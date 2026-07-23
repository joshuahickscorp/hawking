//! The courtyard (Home) digest: fold the append-only event log into the retrospective activity read the
//! launcher shows (session list + stats + heatmap). Every number here is DERIVED from real events; nothing
//! is fabricated. Token totals are deliberately omitted because they are never persisted to the log
//! (GenerationStats is ephemeral), so the launcher simply does not claim a token count.
//!
//! Doctrine: this is a retrospective read, not a budget meter. Totals only, no cap, no remaining percent.

use hide_core::persistence::{DynEventLog, DynProjectionStore};
use hide_core::Result;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const HEATMAP_WEEKS: usize = 18;
const HEATMAP_DAYS: usize = HEATMAP_WEEKS * 7;
const SECS_PER_DAY: u64 = 86_400;
const MAX_SESSIONS: usize = 12;
const TITLE_MAX: usize = 48;

/// One accumulator per session while folding the log.
#[derive(Default)]
struct SessionAgg {
    title: Option<String>,
    updated_us: u64,
    turns: u32,
}

/// Compute the `home` and `sessions` projection patches from the durable event log.
/// Returns `(home_patch, sessions_patch)` as the exact JSON shapes the FE store folds.
pub async fn compute_home_and_sessions(
    event_log: &DynEventLog,
    projection_store: &DynProjectionStore,
    workspace_root: &Path,
) -> Result<(Value, Value)> {
    let events = event_log.scan(None, None, None).await?;
    let now_us = now_micros();
    let today = (now_us / 1_000_000) / SECS_PER_DAY;

    let mut sessions: HashMap<String, SessionAgg> = HashMap::new();
    let mut active_days: BTreeSet<u64> = BTreeSet::new();
    let mut hour_hist = [0u32; 24];
    let mut day_counts: HashMap<u64, u32> = HashMap::new();
    let mut model_tally: HashMap<String, u32> = HashMap::new();
    let mut messages: u64 = 0;

    for ev in &events {
        let secs = ev.ts / 1_000_000;
        let day = secs / SECS_PER_DAY;
        active_days.insert(day);
        *day_counts.entry(day).or_insert(0) += 1;
        hour_hist[((secs % SECS_PER_DAY) / 3_600) as usize % 24] += 1;

        let sid = ev.session_id.to_string();
        let agg = sessions.entry(sid).or_default();
        agg.updated_us = agg.updated_us.max(ev.ts);

        if ev.kind == "user.intent.submit_turn" {
            messages += 1;
            agg.turns += 1;
            if agg.title.is_none() {
                if let Some(text) = ev
                    .payload
                    .get("args")
                    .and_then(|a| a.get("text"))
                    .and_then(|t| t.as_str())
                {
                    agg.title = Some(truncate(text.trim(), TITLE_MAX));
                }
            }
        } else if ev.kind == "runtime.status" {
            if let Some(d) = ev.payload.get("detail").and_then(|v| v.as_str()) {
                let name = model_name(d);
                if !name.is_empty() {
                    *model_tally.entry(name).or_insert(0) += 1;
                }
            }
        }
    }

    let favorite_model = model_tally
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(m, _)| m)
        // Never invent a model. An empty tally means no run ever reported one (a host with no engine
        // is the model-free preview), and the launcher used to print the literal "local" as the
        // favourite model beside a status bar saying no model is configured. Same rule the tokens
        // field below already follows: report what the log holds, or say it is unknown.
        .unwrap_or_else(|| "unknown".to_string());
    let peak_hour = hour_hist
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| **c)
        .map(|(h, _)| h as u32)
        .unwrap_or(0);
    let (streak_current, streak_longest) = streaks(&active_days, today);
    let heatmap = build_heatmap(&day_counts, today);

    let digest = json!({
        "sessions": sessions.len(),
        "messages": messages,
        "active_days": active_days.len(),
        "streak_current": streak_current,
        "streak_longest": streak_longest,
        "peak_hour": peak_hour,
        "favorite_model": favorite_model,
        "heatmap": heatmap,
        "heatmap_cols": HEATMAP_WEEKS,
        // tokens intentionally omitted: not persisted to the event log, so not claimed.
    });

    let home = json!({
        "user": user_json(),
        "workspace": workspace_json(workspace_root),
        "digest": digest,
    });

    // Session rows, most-recently-updated first, capped for the rail.
    let mut rows: Vec<(String, SessionAgg)> = sessions.into_iter().collect();
    rows.sort_by(|a, b| b.1.updated_us.cmp(&a.1.updated_us));
    rows.truncate(MAX_SESSIONS);
    let items: Vec<Value> = rows
        .into_iter()
        .map(|(id, agg)| {
            let state = session_state(projection_store, &id);
            json!({
                "id": id,
                "title": agg.title.unwrap_or_else(|| "session".to_string()),
                "state": state,
                "updated_ms": agg.updated_us / 1_000,
                "turns": agg.turns,
            })
        })
        .collect();

    Ok((home, json!({ "items": items })))
}

fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Current streak (day-tolerant: counts back from today, or from yesterday if today is quiet) and the
/// longest consecutive run anywhere in the history.
fn streaks(days: &BTreeSet<u64>, today: u64) -> (u32, u32) {
    if days.is_empty() {
        return (0, 0);
    }
    // Longest run.
    let mut longest = 1u32;
    let mut run = 1u32;
    let mut prev: Option<u64> = None;
    for &d in days {
        if let Some(p) = prev {
            if d == p + 1 {
                run += 1;
            } else {
                run = 1;
            }
        }
        longest = longest.max(run);
        prev = Some(d);
    }
    // Current run ending at today (tolerate a quiet today by starting from yesterday).
    let mut anchor = if days.contains(&today) {
        today
    } else if days.contains(&today.saturating_sub(1)) {
        today - 1
    } else {
        return (0, longest);
    };
    let mut current = 0u32;
    loop {
        if days.contains(&anchor) {
            current += 1;
            if anchor == 0 {
                break;
            }
            anchor -= 1;
        } else {
            break;
        }
    }
    (current, longest)
}

/// Flat, row-major (col*7 + row) activity counts for the last HEATMAP_DAYS ending today, oldest column
/// first. Matches the FE `heatColumns(counts, cols)` layout.
fn build_heatmap(day_counts: &HashMap<u64, u32>, today: u64) -> Vec<u32> {
    let start = today.saturating_sub((HEATMAP_DAYS - 1) as u64);
    (0..HEATMAP_DAYS as u64)
        .map(|offset| *day_counts.get(&(start + offset)).unwrap_or(&0))
        .collect()
}

fn session_state(store: &DynProjectionStore, session_id: &str) -> &'static str {
    let sid = hide_core::ids::SessionId::from(session_id.to_string());
    match store.latest_projection(&sid) {
        Ok(Some((_, proj))) => match proj.get("phase").and_then(|p| p.as_str()) {
            Some(p) => {
                let p = p.to_ascii_lowercase();
                if p.contains("execut") || p.contains("plan") || p.contains("await") {
                    "active"
                } else if p.contains("fail") {
                    "failed"
                } else if p.contains("done") {
                    "done"
                } else {
                    "idle"
                }
            }
            None => "idle",
        },
        _ => "idle",
    }
}

/// The signed-in identity for the greeting: the OS login name, if available. Omitted (FE falls back to a
/// neutral greeting) when it cannot be read. No fabrication.
fn user_json() -> Value {
    match std::env::var("USER").ok().filter(|s| !s.is_empty()) {
        Some(name) => json!({ "name": name }),
        None => Value::Null,
    }
}

fn workspace_json(root: &Path) -> Value {
    let repo = root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workspace")
        .to_string();
    json!({
        "root": root.to_string_lossy(),
        "repo": repo,
        "branch": git_branch(root),
        "worktrees": worktrees(root),
    })
}

/// Read the current branch from .git/HEAD without shelling out. Detached HEAD or a missing repo yields
/// "main" (a neutral default), never a fabricated branch.
fn git_branch(root: &Path) -> String {
    let head = root.join(".git").join("HEAD");
    match std::fs::read_to_string(head) {
        Ok(s) => s
            .trim()
            .strip_prefix("ref: refs/heads/")
            .map(|b| b.to_string())
            .unwrap_or_else(|| "main".to_string()),
        Err(_) => "main".to_string(),
    }
}

/// The names of any linked git worktrees (from .git/worktrees/<name>), empty when none.
fn worktrees(root: &Path) -> Vec<String> {
    let dir = root.join(".git").join("worktrees");
    match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Trim a model detail like "qwen2.5-7b @ 41 tps" down to the model name.
fn model_name(detail: &str) -> String {
    detail
        .split(" @")
        .next()
        .unwrap_or(detail)
        .split(" (")
        .next()
        .unwrap_or(detail)
        .trim()
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{InMemoryEventLog, NewEvent, UserIntentEvent};
    use hide_core::ids::SessionId;
    use hide_core::persistence::InMemoryProjectionStore;
    use std::sync::Arc;

    async fn submit(log: &DynEventLog, sid: &str, text: &str) {
        let mut ev = NewEvent::user_intent(
            SessionId::from(sid.to_string()),
            UserIntentEvent {
                intent: "submit_turn".to_string(),
                args: json!({ "text": text }),
            },
        );
        ev.kind = "user.intent.submit_turn".to_string();
        log.append(ev).await.unwrap();
    }

    #[tokio::test]
    async fn folds_sessions_messages_and_titles() {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let store: DynProjectionStore = Arc::new(InMemoryProjectionStore::default());
        submit(&log, "ses_a", "fix the pool guard").await;
        submit(&log, "ses_a", "add a regression test").await;
        submit(&log, "ses_b", "port the tokenizer").await;

        let (home, sessions) = compute_home_and_sessions(&log, &store, Path::new("/tmp/hawking"))
            .await
            .unwrap();

        assert_eq!(
            home["digest"]["sessions"],
            json!(2),
            "two distinct sessions"
        );
        assert_eq!(home["digest"]["messages"], json!(3), "three submit_turns");
        // Token total is never claimed (not persisted to the log).
        assert!(
            home["digest"].get("tokens").is_none(),
            "tokens omitted, not faked"
        );
        assert_eq!(home["digest"]["heatmap_cols"], json!(HEATMAP_WEEKS));
        assert_eq!(home["workspace"]["repo"], json!("hawking"));

        let items = sessions["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        let titles: Vec<&str> = items.iter().map(|i| i["title"].as_str().unwrap()).collect();
        assert!(titles.contains(&"fix the pool guard"));
        assert!(titles.contains(&"port the tokenizer"));
        // ses_a has two turns.
        let a = items.iter().find(|i| i["id"] == json!("ses_a")).unwrap();
        assert_eq!(a["turns"], json!(2));
    }

    #[test]
    fn streaks_are_day_tolerant_and_find_the_longest_run() {
        // days 10,11,12 (a run of 3), gap, then 20 (today).
        let days: BTreeSet<u64> = [10u64, 11, 12, 20].into_iter().collect();
        let (current, longest) = streaks(&days, 20);
        assert_eq!(longest, 3);
        assert_eq!(current, 1, "today alone is a current streak of 1");
        // today quiet but yesterday active -> current continues from yesterday.
        let (current2, _) = streaks(&days, 21);
        assert_eq!(current2, 1);
        // two quiet days -> no current streak.
        let (current3, _) = streaks(&days, 22);
        assert_eq!(current3, 0);
    }

    #[test]
    fn heatmap_is_a_full_grid_ending_today() {
        let today = 20_000u64; // a realistic unix-day number (well past the 126-day window)
        let mut counts = HashMap::new();
        counts.insert(today, 5u32);
        let cells = build_heatmap(&counts, today);
        assert_eq!(cells.len(), HEATMAP_DAYS);
        assert_eq!(*cells.last().unwrap(), 5, "today is the last cell");
        assert_eq!(cells[0], 0, "the oldest day is quiet");
    }
}
