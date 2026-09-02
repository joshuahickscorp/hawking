//! Minimal but real HAIDER CLI bootstrap.
//!
//! This is not the final HCLI TUI, but it provides the required user-facing
//! `haider` command surface: project detection, model discovery, runtime health,
//! durable mission state, deterministic tools, and compact evidence output.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::{
    ContextBudgets, ContextGovernor, Haider, MemGate, MissionState, SystemMemGate, TaskType,
    ToolBus, ToolRequest, ToolResult,
};

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let root = detect_project_root()?;
    let mut state = CliState::new(root)?;
    state.print_banner()?;

    let stdin = io::stdin();
    loop {
        print!("haider> ");
        io::stdout().flush()?;
        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let continue_loop = if line.starts_with('/') {
            state.handle_command(line)?
        } else {
            state.handle_natural_language(line)?
        };
        if !continue_loop {
            break;
        }
    }
    state.save()?;
    println!("bye");
    Ok(())
}

struct CliState {
    root: PathBuf,
    mission: MissionState,
    bus: ToolBus,
    haider: Haider,
    governor: ContextGovernor,
}

impl CliState {
    fn new(root: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let mission = MissionState::load(&root);
        let bus = ToolBus::new(root.clone());
        let gate = Arc::new(SystemMemGate::new(3));
        let haider = Haider::new(gate, 3);
        let governor = ContextGovernor::new(131072, 8192);
        Ok(Self {
            root,
            mission,
            bus,
            haider,
            governor,
        })
    }

    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.mission.save(&self.root)?;
        Ok(())
    }

    fn print_banner(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("HAIDER");
        println!("project  {}", self.root.display());
        let runtime = self.runtime_info();
        println!("resident {}", runtime.resident.as_deref().unwrap_or("none"));
        println!("type /help");
        Ok(())
    }

    fn handle_command(&mut self, line: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let mut parts = line.splitn(2, ' ');
        let cmd = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("").trim();
        match cmd {
            "/help" => {
                print_help();
                Ok(true)
            }
            "/exit" | "/quit" => {
                self.save()?;
                Ok(false)
            }
            "/model" => self.cmd_model(rest),
            "/goal" => {
                if !rest.is_empty() {
                    self.mission.set_goal(rest);
                    self.save()?;
                    emit("GOAL", rest);
                }
                Ok(true)
            }
            "/ultragoal" | "/ug" => self.cmd_ultragoal(rest),
            "/steer" => {
                if !rest.is_empty() {
                    self.mission.add_steer(rest);
                    self.save()?;
                    emit("STEER", rest);
                }
                Ok(true)
            }
            "/status" => {
                self.print_status()?;
                Ok(true)
            }
            "/workers" => self.cmd_workers(),
            "/context" => {
                self.print_context_budgets()?;
                Ok(true)
            }
            "/runtime" => self.cmd_runtime(),
            "/frontier" => self.cmd_frontier(),
            "/receipt" => self.cmd_receipt(rest),
            "/why" => self.cmd_why(),
            "/find" => self.cmd_find(rest),
            "/read" => self.cmd_read(rest),
            "/symbols" => self.cmd_symbols(rest),
            "/callers" => self.cmd_callers(rest),
            "/doctor" => self.cmd_doctor(),
            "/gravity" => {
                println!("GRAVITY");
                println!("authority  hawking-core gravity (not duplicated)");
                Ok(true)
            }
            "/odyssey" => {
                println!("ODYSSEY");
                println!("authority  hawking-context skill foundry / odyssey (not duplicated)");
                Ok(true)
            }
            "/bench" => self.cmd_bench(),
            "/budget" => {
                self.print_context_budgets()?;
                self.print_memgate()?;
                Ok(true)
            }
            _ => {
                println!("unknown command: {cmd}");
                Ok(true)
            }
        }
    }

    fn cmd_model(&mut self, rest: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let models = discover_models(&self.root);
        let runtime = self.runtime_info();
        if rest.is_empty() {
            println!("MODEL");
            for (i, m) in models.iter().enumerate() {
                let marker = if self.mission.model.as_deref() == Some(m.name.as_str()) {
                    "●"
                } else {
                    " "
                };
                println!("{marker} {}  {}", i, m.name);
                println!("    {}", m.path.display());
            }
            println!("RESIDENT {}", runtime.resident.as_deref().unwrap_or("none"));
        } else if let Ok(idx) = rest.parse::<usize>() {
            if let Some(m) = models.get(idx) {
                self.mission.model = Some(m.name.clone());
                self.save()?;
                println!("selected {}", m.name);
            }
        } else {
            self.mission.model = Some(rest.to_string());
            self.save()?;
            println!("selected {rest}");
        }
        Ok(true)
    }

    fn cmd_ultragoal(&mut self, rest: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let text = if rest.is_empty() {
            self.read_multiline()?
        } else {
            rest.to_string()
        };
        let summary = self.mission.ingest_text(&text);
        self.save()?;
        println!("ULTRAGOAL INGESTED");
        println!("tokens    ~{}", summary.tokens);
        println!("clauses   {}", summary.clauses);
        println!("obligations {}", summary.obligations);
        Ok(true)
    }

    fn read_multiline(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut text = String::new();
        println!("paste ultragoal, end with a line containing only '.'");
        for line in io::stdin().lock().lines() {
            let l = line?;
            if l.trim() == "." {
                break;
            }
            text.push_str(&l);
            text.push('\n');
        }
        Ok(text)
    }

    fn cmd_workers(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        let res = self.bus.dispatch(ToolRequest {
            tool: "worker.status".into(),
            args: json!({}),
        })?;
        print_tool_result(&res);
        for (i, lane) in self.haider.lanes.lanes.iter().enumerate() {
            let letter = char::from(b'A' + (i % 26) as u8);
            println!("{}", lane.render_line(letter));
        }
        Ok(true)
    }

    fn cmd_runtime(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let r = self.runtime_info();
        println!("RUNTIME");
        println!("endpoint   {}", r.endpoint);
        println!("healthy    {}", r.healthy);
        println!("resident   {}", r.resident.as_deref().unwrap_or("none"));
        for m in &r.models {
            println!("model      {m}");
        }
        Ok(true)
    }

    fn cmd_frontier(&self) -> Result<bool, Box<dyn std::error::Error>> {
        println!("FRONTIER");
        for o in self.mission.frontier() {
            println!("- [{}] {}", format!("{:?}", o.status), o.text);
        }
        Ok(true)
    }

    fn cmd_receipt(&mut self, rest: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let args = if rest.is_empty() {
            json!({})
        } else {
            json!({ "id": rest })
        };
        let res = self.bus.dispatch(ToolRequest {
            tool: "receipt.query".into(),
            args,
        })?;
        print_tool_result(&res);
        Ok(true)
    }

    fn cmd_why(&self) -> Result<bool, Box<dyn std::error::Error>> {
        println!("WHY");
        println!("next     {}", self.mission.next_action().unwrap_or("none"));
        for s in self.mission.steers.iter().rev().take(5) {
            println!("steer    {}", s.text);
        }
        Ok(true)
    }

    fn cmd_find(&mut self, rest: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if rest.is_empty() {
            println!("usage: /find <pattern>");
            return Ok(true);
        }
        let res = self.bus.dispatch(ToolRequest {
            tool: "repo.find".into(),
            args: json!({ "pattern": rest }),
        })?;
        emit("FIND", rest);
        print_tool_result(&res);
        Ok(true)
    }

    fn cmd_read(&mut self, rest: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if rest.is_empty() {
            println!("usage: /read <path>");
            return Ok(true);
        }
        let res = self.bus.dispatch(ToolRequest {
            tool: "repo.read".into(),
            args: json!({ "path": rest }),
        })?;
        emit("READ", rest);
        print_tool_result(&res);
        Ok(true)
    }

    fn cmd_symbols(&mut self, rest: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if rest.is_empty() {
            println!("usage: /symbols <query>");
            return Ok(true);
        }
        let res = self.bus.dispatch(ToolRequest {
            tool: "repo.symbols".into(),
            args: json!({ "query": rest }),
        })?;
        emit("SYMBOLS", rest);
        print_tool_result(&res);
        Ok(true)
    }

    fn cmd_callers(&mut self, rest: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if rest.is_empty() {
            println!("usage: /callers <symbol>");
            return Ok(true);
        }
        let res = self.bus.dispatch(ToolRequest {
            tool: "repo.callers".into(),
            args: json!({ "symbol": rest }),
        })?;
        emit("CALLERS", rest);
        print_tool_result(&res);
        Ok(true)
    }

    fn cmd_doctor(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        let res = self.bus.dispatch(ToolRequest {
            tool: "git.status".into(),
            args: json!({}),
        })?;
        emit("GIT", "status");
        print_tool_result(&res);
        let res = self.bus.dispatch(ToolRequest {
            tool: "shell.run_safe".into(),
            args: json!({ "command": ["cargo", "--version"] }),
        })?;
        emit("RUN", "cargo --version");
        print_tool_result(&res);
        Ok(true)
    }

    fn cmd_bench(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        let res = self.bus.dispatch(ToolRequest {
            tool: "shell.run_safe".into(),
            args: json!({
                "command": ["cargo", "test", "-p", "hide-backend", "--", "haider"],
                "timeout_secs": 300
            }),
        })?;
        emit("BENCH", "cargo test haider");
        print_tool_result(&res);
        Ok(true)
    }

    fn print_status(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("HAIDER");
        println!("PROJECT  {}", self.root.display());
        let runtime = self.runtime_info();
        println!("RESIDENT {}", runtime.resident.as_deref().unwrap_or("none"));
        let p = self.haider.gate.pressure();
        let gb = 1024u64 * 1024 * 1024;
        println!(
            "MEM      {} / {} GB",
            p.wired_bytes.saturating_add(p.compressed_bytes) / gb,
            p.total_physical_bytes / gb
        );
        println!(
            "LANES    {} / {}",
            self.haider.lanes.admitted_count(),
            self.haider.gate.ceiling()
        );
        println!("GOAL     {}", self.mission.goal.as_deref().unwrap_or("none"));
        println!("NEXT     {}", self.mission.next_action().unwrap_or("none"));
        Ok(())
    }

    fn print_context_budgets(&self) -> Result<(), Box<dyn std::error::Error>> {
        let p = self.haider.gate.pressure();
        let pressure = if p.total_physical_bytes == 0 {
            0.0
        } else {
            (p.wired_bytes.saturating_add(p.compressed_bytes)) as f64
                / p.total_physical_bytes as f64
        };
        let b = self.governor.budgets_for(TaskType::Edit, 1000, 1, pressure);
        println!("CONTEXT");
        println!("runtime      {}", self.governor.runtime_context);
        println!("projected    {}", b.projected_context());
        println!("output       {}", b.output_reserve);
        println!(
            "invariant    {}",
            if self.governor.invariant_ok(&b) {
                "ok"
            } else {
                "violated"
            }
        );
        println!("repo_map     {}", b.repo_map);
        println!("task_map     {}", b.task_map);
        println!("source       {}", b.source);
        println!("receipts     {}", b.receipts);
        println!("history      {}", b.history);
        println!("tool_results {}", b.tool_results);
        Ok(())
    }

    fn print_memgate(&self) -> Result<(), Box<dyn std::error::Error>> {
        let p = self.haider.gate.pressure();
        let gb = 1024u64 * 1024 * 1024;
        println!("MEMGATE");
        println!("total      {} GB", p.total_physical_bytes / gb);
        println!("wired      {} GB", p.wired_bytes / gb);
        println!("compressed {} GB", p.compressed_bytes / gb);
        println!("swap       {} GB", p.swap_bytes / gb);
        println!("available  {} GB", p.available_bytes / gb);
        println!(
            "resident   {} GB",
            p.resident_model_bytes / gb
        );
        Ok(())
    }

    fn runtime_info(&self) -> RuntimeInfo {
        let endpoint = self
            .mission
            .endpoint
            .clone()
            .or_else(|| std::env::var("HAIDER_ENDPOINT").ok())
            .unwrap_or_else(|| "http://127.0.0.1:8081".into());
        let mut info = RuntimeInfo {
            endpoint: endpoint.clone(),
            healthy: false,
            models: Vec::new(),
            resident: self.mission.model.clone(),
        };
        if let Ok(v) = http_get_json(&format!("{endpoint}/v1/models")) {
            info.healthy = true;
            if let Some(arr) = v.get("data").and_then(|d| d.as_array()) {
                for m in arr {
                    if let Some(id) = m.get("id").and_then(|x| x.as_str()) {
                        info.models.push(id.to_string());
                    }
                }
            }
            if info.resident.is_none() {
                info.resident = info.models.first().cloned();
            }
        } else {
            let local = discover_models(&self.root);
            info.models = local.iter().map(|m| m.name.clone()).collect();
            if info.resident.is_none() {
                info.resident = info.models.first().cloned();
            }
        }
        info
    }

    fn handle_natural_language(&mut self, text: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if self.mission.goal.is_none() {
            self.mission.set_goal(text);
        } else {
            self.mission.add_steer(text);
        }

        if let Some(path) = extract_path(text, &self.root) {
            emit("READ", &path);
            let res = self.bus.dispatch(ToolRequest {
                tool: "repo.read".into(),
                args: json!({ "path": path }),
            })?;
            print_tool_result(&res);
            let id = write_receipt(&mut self.bus, &format!("read {path}"))?;
            self.mission.add_receipt(&id, &format!("read {path}"));
        } else if let Some(query) = extract_query(text) {
            emit("SEARCH", &query);
            let res = self.bus.dispatch(ToolRequest {
                tool: "repo.search".into(),
                args: json!({ "query": query }),
            })?;
            print_tool_result(&res);
            let id = write_receipt(&mut self.bus, &format!("search {query}"))?;
            self.mission.add_receipt(&id, &format!("search {query}"));
        } else {
            emit("STEER", text);
        }

        self.save()?;
        Ok(true)
    }
}

struct RuntimeInfo {
    endpoint: String,
    healthy: bool,
    models: Vec<String>,
    resident: Option<String>,
}

struct ModelInfo {
    path: PathBuf,
    name: String,
    size_bytes: u64,
}

fn detect_project_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut dir = std::env::current_dir()?;
    for _ in 0..10 {
        if dir.join(".git").exists() || dir.join("Cargo.toml").exists() {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    Ok(std::env::current_dir()?)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn emit(event: &str, detail: &str) {
    println!("→ {event:<8} {}", truncate_single_line(detail, 80));
}

fn print_tool_result(res: &ToolResult) {
    let mark = if res.ok { "✓" } else { "✕" };
    let hash = if res.evidence_hash.len() >= 8 {
        &res.evidence_hash[..8]
    } else {
        &res.evidence_hash
    };
    println!("{mark} {} {}ms hash={hash}", res.tool, res.duration_ms);
    let out = truncate_lines(&res.stdout, 20);
    if !out.is_empty() {
        println!("  {out}");
    }
    let err = truncate_lines(&res.stderr, 10);
    if !err.is_empty() {
        println!("  err: {err}");
    }
}

fn truncate_lines(s: &str, max: usize) -> String {
    let mut lines = s.lines();
    let taken: Vec<&str> = lines.by_ref().take(max).collect();
    let mut out = taken.join("\n  ");
    if lines.next().is_some() {
        out.push_str("\n  ...");
    }
    out
}

fn truncate_single_line(s: &str, max: usize) -> String {
    let s = s.lines().next().unwrap_or("");
    if s.chars().count() > max {
        format!("{}...", s.chars().take(max).collect::<String>())
    } else {
        s.to_string()
    }
}

fn extract_path(text: &str, root: &Path) -> Option<String> {
    for tok in text.split_whitespace() {
        let t = tok.trim_matches(|c| matches!(c, '"' | '\'' | ',' | '.' | ';' | '(' | ')'));
        if t.contains('/')
            || t.contains('\\')
            || t.ends_with(".rs")
            || t.ends_with(".py")
            || t.ends_with(".ts")
            || t.ends_with(".tsx")
            || t.ends_with(".md")
        {
            if Path::new(t).exists() || root.join(t).exists() {
                return Some(t.to_string());
            }
        }
    }
    None
}

fn extract_query(text: &str) -> Option<String> {
    text.split_whitespace()
        .find(|w| w.len() > 2 && !w.starts_with('/'))
        .map(|w| w.trim_matches(|c| matches!(c, '"' | '\'' | ',' | '.' | ';')).to_string())
}

fn write_receipt(bus: &mut ToolBus, summary: &str) -> Result<String, Box<dyn std::error::Error>> {
    let id = format!("R-{}", now_ms());
    let res = bus.dispatch(ToolRequest {
        tool: "receipt.write".into(),
        args: json!({ "id": id, "summary": summary }),
    })?;
    if !res.ok {
        return Err(format!("receipt.write failed: {}", res.stderr).into());
    }
    Ok(id)
}

fn http_get_json(url: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let without = url.strip_prefix("http://").unwrap_or(url);
    let (hostport, path) = without.split_once('/').unwrap_or((without, "/"));
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>()?),
        None => (hostport.to_string(), 80),
    };
    let addr = (host.as_str(), port)
        .to_socket_addrs()?
        .next()
        .ok_or("no socket address")?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(1))?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: {hostport}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf);
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("");
    Ok(serde_json::from_str(body)?)
}

fn discover_models(root: &Path) -> Vec<ModelInfo> {
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("HAIDER_MODEL_DIR") {
        dirs.push(PathBuf::from(d));
    }
    dirs.push(root.join("models"));
    dirs.push(root.join(".haider").join("models"));
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(PathBuf::from(home).join("models"));
        dirs.push(PathBuf::from(home).join(".cache").join("huggingface").join("hub"));
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for d in dirs {
        if d.exists() {
            walk(&d, 0, &mut out, &mut seen);
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<ModelInfo>, seen: &mut BTreeSet<PathBuf>) {
    if depth > 6 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, depth + 1, out, seen);
        } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            if seen.insert(path.clone()) {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                out.push(ModelInfo {
                    path,
                    name,
                    size_bytes: size,
                });
            }
        }
    }
}

fn print_help() {
    println!("HAIDER commands:");
    println!("  /help");
    println!("  /model [index]");
    println!("  /goal <text>");
    println!("  /ultragoal <paste or . terminator>");
    println!("  /steer <text>");
    println!("  /status");
    println!("  /workers");
    println!("  /context");
    println!("  /runtime");
    println!("  /frontier");
    println!("  /receipt [id]");
    println!("  /why");
    println!("  /find <pattern>");
    println!("  /read <path>");
    println!("  /symbols <query>");
    println!("  /callers <symbol>");
    println!("  /doctor");
    println!("  /gravity");
    println!("  /odyssey");
    println!("  /bench");
    println!("  /budget");
    println!("  /exit");
}
