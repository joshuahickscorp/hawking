//! `hawking-gravityd`: native, process-resident Gravity discovery service.
//!
//! It is deliberately a narrow child service for `hawkingd`, not a second
//! daemon authority. The protocol is one JSON request/response per line over
//! stdio so the Python compatibility surface can supervise one long-lived
//! native process without spawning a binary per repository query.

use hawking_core::{
    gravity_repo_search::{RepoSearchLimits, RepoSearchService},
    gravity_tool_catalog::{GravityToolCatalog, GravityToolEntry},
    gravity_tool_dispatch::{ToolDispatchCatalog, ToolDispatchEntry},
    mega_kernel::{MegaKernelPlan, OrganSlot},
    model_lake_catalog::ModelLakeCatalog,
};
use serde_json::{json, Value};
use std::{
    env,
    io::{self, BufRead, Write},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

fn usage() -> &'static str {
    "usage: hawking-gravityd --root <repository> [--max-files N] [--max-file-bytes N]"
}

fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let mut root = None;
    let mut limits = RepoSearchLimits::default();
    while let Some(argument) = arguments.next() {
        match argument.to_string_lossy().as_ref() {
            "--root" => root = arguments.next().map(PathBuf::from),
            "--max-files" => match parse_next_usize(&mut arguments, "--max-files") {
                Ok(value) => limits.max_files = value,
                Err(error) => return fail(error),
            },
            "--max-file-bytes" => match parse_next_u64(&mut arguments, "--max-file-bytes") {
                Ok(value) => limits.max_file_bytes = value,
                Err(error) => return fail(error),
            },
            "--help" | "-h" => {
                println!("{}", usage());
                return ExitCode::SUCCESS;
            }
            _ => return fail(format!("unknown argument {argument:?}; {}", usage())),
        }
    }
    let Some(root) = root else {
        return fail(usage().to_owned());
    };
    let service = match RepoSearchService::open(&root, limits) {
        Ok(service) => service,
        Err(error) => return fail(format!("native Gravity service refused startup: {error}")),
    };
    // This is an already-built metadata catalog, not a ModelLake scan. Keep a
    // coherent read-only generation resident alongside the repository index;
    // a missing catalog stays an explicit query error rather than triggering a
    // filesystem crawl in a tool request.
    let default_catalog_path = root.join("receipts/future/modellake-index/catalog.json");
    let default_catalog = ModelLakeCatalog::load(&default_catalog_path).ok();
    let mut tool_catalog: Option<GravityToolCatalog> = None;
    let mut tool_dispatch: Option<ToolDispatchCatalog> = None;
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => return fail(format!("native Gravity service stdin failure: {error}")),
        };
        let request = match serde_json::from_str::<Value>(&line) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    &mut stdout,
                    json!({"ok": false, "error": format!("invalid JSON: {error}")}),
                );
                continue;
            }
        };
        let operation = request.get("op").and_then(Value::as_str).unwrap_or("");
        if operation == "rank_content_paths" {
            write_rank_content_response(&mut stdout, &service, &request);
            continue;
        }
        let response = match operation {
            "health" => json!({
                "ok": true,
                "schema": "hawking.gravityd.health.v1",
                "generation": service.generation(),
            }),
            "mega_kernel_plan" => mega_kernel_plan_response(&request),
            "search" => {
                let needle = request.get("needle").and_then(Value::as_str).unwrap_or("");
                let max_results = request
                    .get("max_results")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(limits.max_results);
                match service.search(needle, max_results) {
                    Ok(result) => json!({
                        "ok": true,
                        "schema": "hawking.gravityd.search.v1",
                        "generation": service.generation(),
                        "files_seen": result.files_seen,
                        "skipped_large": result.skipped_large,
                        "truncated": result.truncated,
                        "matches": result.matches.into_iter().map(|item| json!({
                            "path": item.path,
                            "line": item.line,
                            "text": item.text,
                        })).collect::<Vec<_>>(),
                    }),
                    Err(error) => json!({"ok": false, "error": error.to_string()}),
                }
            }
            "search_files" => {
                let requested = request
                    .get("path")
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| root.clone());
                let needle = request.get("needle").and_then(Value::as_str).unwrap_or("");
                let glob = request.get("glob").and_then(Value::as_str).unwrap_or("*");
                let max_results = request
                    .get("max_results")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(100);
                let max_per_file = request
                    .get("max_per_file")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok());
                let started = Instant::now();
                match service.search_files(&requested, needle, glob, max_results, max_per_file) {
                    Ok(result) => json!({
                        "ok": true,
                        "schema": "hawking.gravityd.search-files.v1",
                        "generation": service.generation(),
                        "freshness": "metadata-validated",
                        "root": requested,
                        "pattern": needle,
                        "files_seen": result.files_seen,
                        "skipped_large": result.skipped_large,
                        "truncated": result.truncated,
                        "matches": result.matches.into_iter().map(|item| json!({
                            "path": item.path,
                            "line": item.line,
                            "text": item.text,
                        })).collect::<Vec<_>>(),
                        "elapsed_ns": started.elapsed().as_nanos(),
                    }),
                    Err(error) => json!({"ok": false, "error": error.to_string()}),
                }
            }
            "list" => {
                let requested = request
                    .get("path")
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| root.clone());
                let glob = request.get("glob").and_then(Value::as_str).unwrap_or("*");
                let max_results = request
                    .get("max_results")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(500);
                let recursive = request
                    .get("recursive")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                match service.list(&requested, glob, max_results, recursive) {
                    Ok(result) => json!({
                        "ok": true,
                        "schema": "hawking.gravityd.list.v1",
                        "root": requested,
                        "glob": glob,
                        "files": result.files.into_iter().map(list_entry_json).collect::<Vec<_>>(),
                        "directories": result.directories.into_iter().map(list_entry_json).collect::<Vec<_>>(),
                        "truncated": result.truncated,
                        "directories_seen": result.directories_seen,
                        "elapsed_ns": result.elapsed_ns,
                    }),
                    Err(error) => json!({"ok": false, "error": error.to_string()}),
                }
            }
            "read_file" => read_file_response(&root, &request),
            "rank_code_paths" => {
                let suffixes = request
                    .get("suffixes")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let words = request
                    .get("words")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let max_results = request
                    .get("max_results")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(4);
                match service.rank_code_paths(&suffixes, &words, max_results) {
                    Ok(result) => json!({
                        "ok": true,
                        "schema": "hawking.gravityd.rank-code-paths.v1",
                        "generation": service.generation(),
                        "path_generation": service.path_generation(),
                        "paths": result.paths,
                        "files_seen": result.files_seen,
                        "complete": result.complete,
                        "elapsed_ns": result.elapsed_ns,
                    }),
                    Err(error) => json!({"ok": false, "error": error.to_string()}),
                }
            }
            "rebuild" => match service.rebuild() {
                Ok(rebuilt) => json!({
                    "ok": true,
                    "schema": "hawking.gravityd.rebuild.v1",
                    "generation": rebuilt.generation,
                    "indexed_files": rebuilt.indexed_files,
                    "elapsed_ns": rebuilt.elapsed_ns,
                }),
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            },
            "catalog" => catalog_response(&root, default_catalog.as_ref(), &request),
            "tool_catalog_load" => match parse_tool_catalog(&request) {
                Ok(catalog) => {
                    let count = catalog.len();
                    tool_catalog = Some(catalog);
                    json!({"ok": true, "schema": "hawking.gravityd.tool-catalog-load.v1", "count": count})
                }
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            },
            "tool_catalog_query" => match tool_catalog_query(tool_catalog.as_ref(), &request) {
                Ok(response) => response,
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            },
            "tool_dispatch_load" => match parse_tool_dispatch(&request) {
                Ok(catalog) => {
                    let count = catalog.len();
                    tool_dispatch = Some(catalog);
                    json!({"ok": true, "schema": "hawking.gravityd.tool-dispatch-load.v1", "count": count})
                }
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            },
            "tool_dispatch_admit" => match tool_dispatch_admit(tool_dispatch.as_ref(), &request) {
                Ok(response) => response,
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            },
            "shutdown" => {
                write_response(
                    &mut stdout,
                    json!({"ok": true, "schema": "hawking.gravityd.shutdown.v1"}),
                );
                return ExitCode::SUCCESS;
            }
            _ => {
                json!({"ok": false, "error": "op must be health, list, read_file, rank_code_paths, rank_content_paths, search, search_files, rebuild, catalog, tool_catalog_load, tool_catalog_query, tool_dispatch_load, tool_dispatch_admit, or shutdown"})
            }
        };
        write_response(&mut stdout, response);
    }
    ExitCode::SUCCESS
}

/// Serve the Rust-owned MegaKernel contract without selecting a backend.
///
/// The daemon does not inspect weights or synthesize anatomy here: the caller
/// provides its already-recognized model/organ metadata and the core merely
/// normalizes it into the shared, deterministic plan.  This makes the native
/// lowering registry observable to HCLI without creating a second runtime or
/// turning a plan into an execution claim.
fn mega_kernel_plan_response(request: &Value) -> Value {
    let model_id = request
        .get("model_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("unknown");
    let mut plan = MegaKernelPlan::new(model_id);
    if let Some(organs) = request.get("organs").and_then(Value::as_array) {
        for (index, organ) in organs.iter().enumerate() {
            let Some(organ) = organ.as_object() else {
                continue;
            };
            let organ_id = organ
                .get("organ_id")
                .or_else(|| organ.get("id"))
                .or_else(|| organ.get("name"))
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("organ-{index}"));
            let field = |name: &str, fallback: &str| {
                organ
                    .get(name)
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or(fallback)
                    .to_owned()
            };
            plan.upsert_organ(OrganSlot {
                organ_id,
                operation: field("operation", "unknown"),
                representation: field("representation", "model_declared"),
                state: field("state", "model_declared"),
                route: field("route", "dynamic_if_applicable"),
            });
        }
    }
    json!({
        "ok": true,
        "schema": "hawking.gravityd.mega-kernel-plan.v1",
        "plan": plan,
        "claim_boundary": "plan and registered-lowering evidence only; no backend selection, execution, parity, capability, or throughput claim",
    })
}

fn list_entry_json(entry: hawking_core::gravity_repo_search::RepoListEntry) -> Value {
    json!({
        "path": entry.path,
        "filename": entry.filename,
        "type": entry.entry_type,
        "kind": entry.entry_type,
        "size": entry.size_bytes,
        "bytes": entry.size_bytes,
    })
}

fn read_file_response(root: &PathBuf, request: &Value) -> Value {
    let requested = request
        .get("path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.clone());
    let max_bytes = request
        .get("max_bytes")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(64 * 1024);
    let start_line = request
        .get("start_line")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let end_line = request
        .get("end_line")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let started = Instant::now();
    match hawking_core::gravity_repo_search::read_repo_file(
        root, requested, max_bytes, start_line, end_line,
    ) {
        Ok(result) => {
            let path = result.path.clone();
            let sha256 = result.sha256.clone();
            let bytes = result.bytes;
            let mut response = serde_json::to_value(result)
                .unwrap_or_else(|error| json!({"ok": false, "error": error.to_string()}));
            if let Some(object) = response.as_object_mut() {
                object.insert("ok".into(), json!(true));
                object.insert("schema".into(), json!("hawking.gravityd.read-file.v1"));
                object.insert("timing_unit".into(), json!("ns"));
                object.insert("elapsed_ns".into(), json!(started.elapsed().as_nanos()));
                object.insert(
                    "artifact".into(),
                    json!({
                        "kind": "file",
                        "path": path,
                        "sha256": sha256,
                        "bytes": bytes,
                    }),
                );
                object.insert("provenance".into(), json!("hawking-gravityd.read-file.v1"));
            }
            response
        }
        Err(error) => json!({
            "ok": false,
            "schema": "hawking.gravityd.read-file.v1",
            "error": error.to_string(),
            "timing_unit": "ns",
            "elapsed_ns": started.elapsed().as_nanos(),
        }),
    }
}

fn parse_tool_dispatch(request: &Value) -> hawking_core::Result<ToolDispatchCatalog> {
    let rows = request
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            hawking_core::Error::Gravity("tool_dispatch_load requires entries array".into())
        })?;
    let entries = rows
        .iter()
        .map(|row| {
            let name = row.get("name").and_then(Value::as_str).ok_or_else(|| {
                hawking_core::Error::Gravity("tool dispatch entry name missing".into())
            })?;
            let mutation = row.get("mutation").and_then(Value::as_str).ok_or_else(|| {
                hawking_core::Error::Gravity(format!("tool dispatch entry {name} mutation missing"))
            })?;
            Ok(ToolDispatchEntry {
                name: name.to_owned(),
                mutation: mutation.to_owned(),
            })
        })
        .collect::<hawking_core::Result<Vec<_>>>()?;
    ToolDispatchCatalog::from_entries(entries)
}

fn tool_dispatch_admit(
    catalog: Option<&ToolDispatchCatalog>,
    request: &Value,
) -> hawking_core::Result<Value> {
    let catalog = catalog.ok_or_else(|| {
        hawking_core::Error::Gravity("tool dispatch catalog has not been loaded".into())
    })?;
    let name = request
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| hawking_core::Error::Gravity("tool_dispatch_admit requires name".into()))?;
    let mutation = request
        .get("mutation")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            hawking_core::Error::Gravity("tool_dispatch_admit requires mutation".into())
        })?;
    let permissions = request
        .get("permissions")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            hawking_core::Error::Gravity("tool_dispatch_admit requires permissions array".into())
        })?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                hawking_core::Error::Gravity("tool dispatch permissions must be strings".into())
            })
        })
        .collect::<hawking_core::Result<Vec<_>>>()?;
    let started = Instant::now();
    let admission = catalog.admit(name, mutation, &permissions);
    Ok(json!({
        "ok": true,
        "schema": "hawking.gravityd.tool-dispatch-admit.v1",
        "admitted": admission.admitted,
        "mutation": admission.mutation,
        "reason": admission.reason,
        "elapsed_ns": started.elapsed().as_nanos(),
    }))
}

fn parse_tool_catalog(request: &Value) -> hawking_core::Result<GravityToolCatalog> {
    let rows = request
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            hawking_core::Error::Gravity("tool_catalog_load requires entries array".into())
        })?;
    let entries = rows
        .iter()
        .map(|row| {
            let name = row.get("name").and_then(Value::as_str).ok_or_else(|| {
                hawking_core::Error::Gravity("tool catalog entry name missing".into())
            })?;
            let search_text = row
                .get("search_text")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    hawking_core::Error::Gravity(format!(
                        "tool catalog entry {name} search_text missing"
                    ))
                })?;
            Ok(GravityToolEntry {
                name: name.to_owned(),
                search_text: search_text.to_owned(),
            })
        })
        .collect::<hawking_core::Result<Vec<_>>>()?;
    GravityToolCatalog::from_entries(entries)
}

fn tool_catalog_query(
    catalog: Option<&GravityToolCatalog>,
    request: &Value,
) -> hawking_core::Result<Value> {
    let catalog = catalog
        .ok_or_else(|| hawking_core::Error::Gravity("tool catalog has not been loaded".into()))?;
    let terms = request
        .get("terms")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            hawking_core::Error::Gravity("tool_catalog_query requires terms array".into())
        })?
        .iter()
        .map(|term| {
            term.as_str().map(str::to_owned).ok_or_else(|| {
                hawking_core::Error::Gravity("tool catalog terms must be strings".into())
            })
        })
        .collect::<hawking_core::Result<Vec<_>>>()?;
    let max_results = request
        .get("max_results")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(12);
    let started = Instant::now();
    let (names, match_count) = catalog.query(&terms, max_results)?;
    Ok(
        json!({"ok": true, "schema": "hawking.gravityd.tool-catalog-query.v1", "names": names, "match_count": match_count, "elapsed_ns": started.elapsed().as_nanos()}),
    )
}

/// Query the already-built ModelLake catalog only. This deliberately refuses
/// to substitute a live filesystem walk when the catalog is missing.
fn catalog_response(
    root: &PathBuf,
    resident_catalog: Option<&ModelLakeCatalog>,
    request: &Value,
) -> Value {
    let started = Instant::now();
    let explicit = request
        .get("catalog")
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let loaded;
    let catalog = if let Some(path) = explicit {
        loaded = match ModelLakeCatalog::load(path) {
            Ok(catalog) => catalog,
            Err(error) => return json!({"ok": false, "error": error.to_string()}),
        };
        &loaded
    } else if let Some(catalog) = resident_catalog {
        catalog
    } else {
        let path = root.join("receipts/future/modellake-index/catalog.json");
        return json!({"ok": false, "error": format!("native ModelLake catalog is unavailable at {}", path.display())});
    };
    let summary = json!({
        "path": catalog.path(),
        "specimen_count": catalog.specimen_count,
        "partial_count": catalog.partial_count,
        "used_bytes": catalog.used_bytes,
        "budget_bytes": catalog.budget_bytes,
        "over_budget": catalog.over_budget,
    });
    let Some(slug) = request.get("slug").and_then(Value::as_str) else {
        return json!({"ok": true, "schema": "hawking.gravityd.catalog.v1", "elapsed_ns": started.elapsed().as_nanos(), "catalog": summary});
    };
    match catalog.specimen(slug) {
        Ok(Some(row)) => json!({
            "ok": true,
            "schema": "hawking.gravityd.catalog.v1",
            "elapsed_ns": started.elapsed().as_nanos(),
            "catalog": summary,
            "specimen": {
                "slug": row.slug,
                "repo": row.repository,
                "revision": row.revision,
                "bytes": row.bytes,
                "architecture_family": row.architecture_family,
                "lifecycle": row.lifecycle,
                "seal_status": row.seal_status,
                "location": row.location,
            },
        }),
        Ok(None) => {
            json!({"ok": false, "error": format!("ModelLake slug {slug:?} is absent from the catalog")})
        }
        Err(error) => json!({"ok": false, "error": error.to_string()}),
    }
}

#[derive(serde::Serialize)]
struct RankContentResponse {
    ok: bool,
    schema: &'static str,
    generation: u64,
    paths: Vec<PathBuf>,
    files_seen: usize,
    terms_used: usize,
    elapsed_ns: u128,
}

fn write_rank_content_response(
    stdout: &mut impl Write,
    service: &RepoSearchService,
    request: &Value,
) {
    let terms = request
        .get("terms")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let scope_prefix = request
        .get("scope_prefix")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    let test_focus = request
        .get("test_focus")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let max_results = request
        .get("max_results")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(4);
    match service.rank_content_paths(&terms, scope_prefix, test_focus, max_results) {
        Ok(result) => write_serialized_response(
            stdout,
            &RankContentResponse {
                ok: true,
                schema: "hawking.gravityd.rank-content-paths.v1",
                generation: service.generation(),
                paths: result.paths,
                files_seen: result.files_seen,
                terms_used: result.terms_used,
                elapsed_ns: result.elapsed_ns,
            },
        ),
        Err(error) => write_response(stdout, json!({"ok": false, "error": error.to_string()})),
    }
}

fn parse_next_usize(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<usize, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?
        .to_string_lossy()
        .parse()
        .map_err(|_| format!("{flag} must be a positive integer"))
}

fn parse_next_u64(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<u64, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?
        .to_string_lossy()
        .parse()
        .map_err(|_| format!("{flag} must be a positive integer"))
}

fn write_response(stdout: &mut impl Write, response: Value) {
    write_serialized_response(stdout, &response);
}

fn write_serialized_response(stdout: &mut impl Write, response: &impl serde::Serialize) {
    serde_json::to_writer(&mut *stdout, response).expect("JSON response serialization cannot fail");
    writeln!(stdout).expect("native Gravity service stdout is unavailable");
    stdout
        .flush()
        .expect("native Gravity service stdout flush failed");
}

fn fail(message: String) -> ExitCode {
    eprintln!("{message}");
    ExitCode::from(2)
}
