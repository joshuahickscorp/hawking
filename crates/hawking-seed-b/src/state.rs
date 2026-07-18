//! One state machine / transition engine. `(from, event) -> to` guarded by a single table; every
//! transition is a sealed Record appended to an append-only JSONL log, so crash/drain/resume is one
//! mechanism (re-open = replay + verify every seal). Reused from Candidate A verbatim.

use crate::record::Record;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Idle,
    Prepared,
    Admitted,
    Running,
    Draining,
    Paused,
    Blocked,
    Failed,
    Sealed,
}
impl State {
    pub fn as_str(&self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Prepared => "prepared",
            State::Admitted => "admitted",
            State::Running => "running",
            State::Draining => "draining",
            State::Paused => "paused",
            State::Blocked => "blocked",
            State::Failed => "failed",
            State::Sealed => "sealed",
        }
    }
    pub fn from_str(s: &str) -> Option<State> {
        Some(match s {
            "idle" => State::Idle,
            "prepared" => State::Prepared,
            "admitted" => State::Admitted,
            "running" => State::Running,
            "draining" => State::Draining,
            "paused" => State::Paused,
            "blocked" => State::Blocked,
            "failed" => State::Failed,
            "sealed" => State::Sealed,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Prepare,
    Admit,
    Run,
    Evaluate,
    Seal,
    Drain,
    Pause,
    Resume,
    Fail,
}

/// The single transition table. Unknown (state, event) pairs are rejected — no procedural drift.
fn transition(from: State, ev: Event) -> Option<State> {
    use Event::*;
    use State::*;
    Some(match (from, ev) {
        (Idle, Prepare) => Prepared,
        (Prepared, Admit) => Admitted,
        (Admitted, Run) => Running,
        (Running, Evaluate) => Running,
        (Running, Seal) => Sealed,
        (Running, Drain) => Draining,
        (Running, Pause) => Paused,
        (Running, Fail) => Failed,
        (Draining, Seal) => Sealed,
        (Paused, Resume) => Running,
        (Failed, Resume) => Running,
        _ => return None,
    })
}

/// The controller: one state, an append-only Record log, one persistence root.
pub struct Machine {
    pub state: State,
    pub root: PathBuf,
    pub log: Vec<Record>,
}

impl Machine {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        let log_path = root.join("log.jsonl");
        let (state, log) = if log_path.exists() {
            let mut log = Vec::new();
            let mut state = State::Idle;
            for line in std::fs::read_to_string(&log_path)?.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let r: Record = serde_json::from_str(line)?;
                r.verify()?; // resume only from sealed, untampered records
                if let Some(s) = State::from_str(&r.state) {
                    state = s;
                }
                log.push(r);
            }
            (state, log)
        } else {
            (State::Idle, Vec::new())
        };
        Ok(Machine { state, root, log })
    }

    /// Apply one event: guard via the transition table, record a sealed transition receipt, persist.
    pub fn apply(&mut self, ev: Event, payload: serde_json::Value) -> Result<State> {
        let to = transition(self.state, ev).ok_or_else(|| {
            Error::Transition(format!("no transition {:?} --{:?}-->", self.state, ev))
        })?;
        let rec = Record::new(
            "transition",
            serde_json::json!({"from": self.state.as_str(), "event": format!("{ev:?}"), "payload": payload}),
        )
        .with_state(to.as_str())
        .sealed();
        self.append(&rec)?;
        self.state = to;
        Ok(to)
    }

    fn append(&mut self, rec: &Record) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("log.jsonl"))?;
        writeln!(f, "{}", serde_json::to_string(rec)?)?;
        self.log.push(rec.clone());
        Ok(())
    }

    /// Persist an arbitrary sealed evidence record into the same log.
    pub fn record(&mut self, rec: Record) -> Result<()> {
        rec.verify()?;
        self.append(&rec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("seedb-state-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn happy_path_and_crash_resume() {
        let root = scratch("hp");
        {
            let mut m = Machine::open(&root).unwrap();
            for ev in [Event::Prepare, Event::Admit, Event::Run] {
                m.apply(ev, serde_json::json!({})).unwrap();
            }
            assert_eq!(m.state, State::Running);
        }
        // re-open (crash/resume): replay + verify seals, land back in Running
        let m2 = Machine::open(&root).unwrap();
        assert_eq!(m2.state, State::Running);
        assert!(m2.log.iter().all(|r| r.verify().is_ok()));
    }

    #[test]
    fn drain_then_seal() {
        let root = scratch("dr");
        let mut m = Machine::open(&root).unwrap();
        for ev in [Event::Prepare, Event::Admit, Event::Run, Event::Drain, Event::Seal] {
            m.apply(ev, serde_json::json!({})).unwrap();
        }
        assert_eq!(m.state, State::Sealed);
    }

    #[test]
    fn unknown_transition_rejected() {
        let root = scratch("bad");
        let mut m = Machine::open(&root).unwrap();
        assert!(m.apply(Event::Run, serde_json::json!({})).is_err());
    }
}
