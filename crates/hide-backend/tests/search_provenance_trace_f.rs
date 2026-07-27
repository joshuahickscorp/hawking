//! Campaign Trace F (search), section 23, model-free, headless.
//!
//! One end-to-end proof, over the REAL [`BackendHost`], that the five search
//! surfaces reach the same durable substrate and every result is bounded + typed:
//!
//!   1. Search a SYMBOL / FILE through the real `search.text` tool (the code
//!      search path) via `host.dispatch_tool`; the hit is `{path, line, text}` and
//!      the tool result is durably logged.
//!   2. Search a related THREAD / TRANSCRIPT item through the live `run_search`
//!      custom arm over `/intent` (literal + a structured `role` filter).
//!   3. Search a verification RECEIPT: seal one with `run_static_analysis`, then
//!      locate it via `verification_receipts` (a receipt search over `scope`).
//!   4. ATTACH a chosen search result to the current turn as a durable typed link
//!      event (`context.attach`, tagged with the turn's run id).
//!   5. Open PROVENANCE: resolve the attached links back to the real source event
//!      id (transcript) and the path + line (code) they came from.
//!
//! Everything is durable (re-scanned, and re-opened over the same workspace) and
//! bounded (every search caps its result count). No model, no network.

use hide_backend::{BackendHost, BackendServices, EvidenceLink, TranscriptQuery};
use hide_core::api::{Intent, UiEvent, UiEventKind};
use hide_core::config::HideConfig;
use hide_core::event::{Event, NewEvent};
use hide_core::ids::{now_ms, RunId, SessionId};
use hide_core::tool::{ToolCall, ToolStatus};
use hide_core::types::Decision;
use hide_verify::SourceFile;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_dir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("hide_trace_f_{}_{}", now_ms(), uniq))
}

async fn scan(host: &BackendHost, session: &SessionId) -> Vec<Event> {
    host.services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap()
}

/// The first `Custom` UiEvent on the bus whose `kind` field matches, or a panic on
/// timeout (mirrors the wave3 harness).
async fn next_custom(
    rx: &mut tokio::sync::broadcast::Receiver<UiEvent>,
    kind: &str,
) -> serde_json::Value {
    loop {
        let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should arrive")
            .expect("broadcast delivers");
        if let UiEventKind::Custom(value) = ev.kind {
            if value.get("kind").and_then(|k| k.as_str()) == Some(kind) {
                return value;
            }
        }
    }
}

#[tokio::test]
async fn trace_f_search_symbol_thread_receipt_attach_and_open_provenance() {
    // Distinct tokens so each search surface matches only its own seed.
    const CODE_SYMBOL: &str = "parse_port_zzsym";
    const THREAD_TOKEN: &str = "ZZTHREADTOKEN";

    let dir = unique_dir();
    let session: SessionId;
    let run = RunId::new();
    let code_hit_path: String;
    let code_hit_line: u32;
    let transcript_event_id: String;
    let verification_id: String;

    {
        // A trusted headless workspace: the `search.text` code tool advertises the
        // `index.read` capability, which the default policy does not rule on, so a
        // trusted-workspace default_decision admits it (mirrors how Trace C admits
        // writes). The re-open below uses the plain default host.
        let mut config = HideConfig::for_workspace(&dir);
        config.security.default_decision = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        session = host.services.session();
        let log = host.services.event_log.clone();
        let mut rx = host.subscribe_ui();

        // --- (1) SYMBOL / FILE search over the real code search tool -----------
        // Write a fixture into the workspace and search it with `search.text`
        // through the host dispatch path (durably logged as a tool.result).
        let code_dir = dir.join("src");
        std::fs::create_dir_all(&code_dir).unwrap();
        let fixture_path = code_dir.join("net.rs");
        let fixture = format!(
            "pub fn {CODE_SYMBOL}(raw: &str) -> u16 {{\n    raw.parse::<u16>().unwrap_or(0)\n}}\n"
        );
        std::fs::write(&fixture_path, &fixture).unwrap();

        let result = host
            .dispatch_tool(
                session.clone(),
                Some(run.clone()),
                ToolCall::new(
                    "search.text",
                    json!({
                        "pattern": format!("fn {CODE_SYMBOL}"),
                        "root": code_dir.to_string_lossy(),
                        "max": 20
                    }),
                ),
            )
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok, "code search runs: {result:?}");
        let body = result.structured_content.clone().unwrap();
        let matches = body["matches"].as_array().unwrap();
        // Bounded + typed: one precise `{path, line, text}` hit, not truncated.
        assert_eq!(matches.len(), 1, "the symbol matches exactly once: {body}");
        assert_eq!(body["truncated"].as_bool(), Some(false), "result is bounded");
        code_hit_path = matches[0]["path"].as_str().unwrap().to_string();
        code_hit_line = matches[0]["line"].as_u64().unwrap() as u32;
        assert!(code_hit_path.ends_with("net.rs"));
        assert_eq!(code_hit_line, 1, "the symbol is on line 1");

        // The code search is DURABLE: a tool.result carrying the hit is on the log.
        let after_search = scan(&host, &session).await;
        assert!(
            after_search
                .iter()
                .any(|e| e.kind == "tool.result"
                    && serde_json::to_string(&e.payload).unwrap().contains(CODE_SYMBOL)),
            "the code search hit is durably logged as a tool.result"
        );

        // --- (2) THREAD / TRANSCRIPT search over the live run_search arm -------
        // Seed a small transcript: an assistant and a user item both carry the
        // token, so a literal search hits both and a role filter narrows it.
        for (role, text) in [
            ("assistant", format!("{CODE_SYMBOL} referenced in {THREAD_TOKEN} boot path")),
            ("user", format!("where is {THREAD_TOKEN} discussed")),
        ] {
            log.append(NewEvent::system(
                session.clone(),
                "agent.message",
                json!({ "role": role, "text": text }),
            ))
            .await
            .unwrap();
        }

        // (a) LITERAL search dialed with the FE's registered custom name.
        let ack = host
            .handle_intent(Intent::Custom {
                name: "run_search".to_string(),
                payload: json!({ "query": THREAD_TOKEN, "limit": 10 }),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "the search intent is accepted");
        let results = next_custom(&mut rx, "search_results").await;
        assert_eq!(results["query"].as_str(), Some(THREAD_TOKEN));
        let count = results["count"].as_u64().unwrap();
        assert_eq!(count, 2, "literal search hits both items: {results}");
        assert!(count <= 10, "the result honors the caller's limit (bounded)");

        // (b) STRUCTURED filter (role) narrows to the assistant item.
        let ack = host
            .handle_intent(Intent::Custom {
                name: "run_search".to_string(),
                payload: json!({ "query": THREAD_TOKEN, "role": "assistant" }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let filtered = next_custom(&mut rx, "search_results").await;
        assert_eq!(
            filtered["count"].as_u64(),
            Some(1),
            "the role filter narrows to the assistant item: {filtered}"
        );
        assert_eq!(filtered["hits"][0]["role"].as_str(), Some("assistant"));

        // The typed hit (with provenance) that we will attach: take it straight
        // from the host search method so we get a `TranscriptHit` (not JSON).
        let hits = host
            .search_transcript(
                &TranscriptQuery::literal(THREAD_TOKEN)
                    .in_session(session.clone())
                    .with_role("assistant"),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "one assistant hit to cite");
        transcript_event_id = hits[0].event_id.as_str().to_string();
        let transcript_link = EvidenceLink::from_hit(&hits[0]);
        assert_eq!(transcript_link.event_id.as_deref(), Some(transcript_event_id.as_str()));

        // --- (3) RECEIPT search: seal one, then locate it -----------------------
        let receipt = host
            .run_static_analysis(
                session.clone(),
                vec![SourceFile::new("src/net.rs", &fixture)],
            )
            .await
            .unwrap();
        verification_id = receipt.receipt.verification_id.clone();

        // Locate the sealed receipt via the reader (a receipt search over scope).
        let receipts = host.verification_receipts(&session).await.unwrap();
        let found: Vec<_> = receipts
            .iter()
            .filter(|r| r.receipt.scope.iter().any(|p| p == "src/net.rs"))
            .collect();
        assert_eq!(found.len(), 1, "exactly one receipt scoped to the file");
        assert_eq!(
            found[0].receipt.verification_id, verification_id,
            "the located receipt is the one we sealed"
        );

        // --- (4) ATTACH the search results to the CURRENT TURN ------------------
        // A durable typed link event, tagged with the turn's run id. The code hit
        // carries path + line; the transcript hit carries session + event id.
        let code_link = EvidenceLink {
            path: Some(code_hit_path.clone()),
            line: Some(code_hit_line),
            snippet: Some(format!("fn {CODE_SYMBOL}")),
            ..EvidenceLink::default()
        };
        let links = vec![transcript_link.clone(), code_link.clone()];
        log.append(
            NewEvent::system(
                session.clone(),
                "context.attach",
                json!({
                    "attached_to_run": run.as_str(),
                    "verification_id": verification_id,
                    "links": links,
                }),
            )
            .with_run(run.clone()),
        )
        .await
        .unwrap();

        // --- (5) OPEN PROVENANCE from the attachment (still same host) ---------
        assert_provenance_resolves(
            &host,
            &session,
            &run,
            &transcript_event_id,
            &code_hit_path,
            code_hit_line,
            CODE_SYMBOL,
        )
        .await;
    }

    // --- DURABILITY: re-open the workspace and prove it all survived -----------
    let reopened = BackendHost::open_workspace(&dir).unwrap();
    // The receipt survives (readable by the same reader after restart).
    let receipts = reopened.verification_receipts(&session).await.unwrap();
    assert!(
        receipts.iter().any(|r| r.receipt.verification_id == verification_id),
        "the sealed receipt survives a workspace re-open"
    );
    // The transcript search still resolves the same hit after restart.
    let hits = reopened
        .search_transcript(&TranscriptQuery::literal(THREAD_TOKEN).in_session(session.clone()))
        .await
        .unwrap();
    assert!(
        hits.iter().any(|h| h.event_id.as_str() == transcript_event_id),
        "the cited transcript hit survives a workspace re-open"
    );
    // The attachment + its provenance survive the restart end to end.
    assert_provenance_resolves(
        &reopened,
        &session,
        &run,
        &transcript_event_id,
        &code_hit_path,
        code_hit_line,
        CODE_SYMBOL,
    )
    .await;
}

/// Resolve the durable `context.attach` event back to its sources: the transcript
/// link's `event_id` must name a real event on the log, and the code link's
/// `path` + `line` must name the fixture line that still holds the symbol. Bounded
/// and typed throughout (the payload deserializes to `EvidenceLink`s).
#[allow(clippy::too_many_arguments)]
async fn assert_provenance_resolves(
    host: &BackendHost,
    session: &SessionId,
    run: &RunId,
    transcript_event_id: &str,
    code_hit_path: &str,
    code_hit_line: u32,
    code_symbol: &str,
) {
    let events = scan(host, session).await;

    // The attachment is durable and tied to the current turn's run id.
    let attach = events
        .iter()
        .find(|e| e.kind == "context.attach")
        .expect("the attachment survives on the durable log");
    assert_eq!(
        attach.payload["attached_to_run"].as_str(),
        Some(run.as_str()),
        "the attachment is bound to the current turn"
    );

    // Typed: the payload is a bounded list of EvidenceLinks.
    let links: Vec<EvidenceLink> =
        serde_json::from_value(attach.payload["links"].clone()).expect("links are typed");
    assert_eq!(links.len(), 2, "a bounded, typed attachment (two cited links)");

    // Provenance (transcript): the cited event id resolves to a REAL log event.
    let transcript_link = links
        .iter()
        .find(|l| l.event_id.is_some())
        .expect("a transcript link");
    let cited = transcript_link.event_id.as_deref().unwrap();
    assert_eq!(cited, transcript_event_id);
    assert!(
        events.iter().any(|e| e.id.as_str() == cited),
        "the attached transcript hit resolves to a real source event id"
    );

    // Provenance (code): the cited path + line resolves to the fixture line that
    // still holds the symbol on disk.
    let code_link = links
        .iter()
        .find(|l| l.path.is_some())
        .expect("a code link");
    assert_eq!(code_link.path.as_deref(), Some(code_hit_path));
    assert_eq!(code_link.line, Some(code_hit_line));
    let text = std::fs::read_to_string(code_hit_path).expect("the cited file is readable");
    let line = text
        .lines()
        .nth((code_hit_line - 1) as usize)
        .expect("the cited line exists");
    assert!(
        line.contains(code_symbol),
        "the cited path+line still holds the symbol (open-provenance jump target)"
    );
}
