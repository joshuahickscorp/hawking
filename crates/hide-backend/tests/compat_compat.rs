use hide_backend::compat::settings::{Decision, Scope};
use hide_backend::compat::{claude_md, mcp, settings, CompatConfig, Layout};
use std::fs;
use std::path::{Path, PathBuf};
fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}
struct Fixture {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    home: PathBuf,
    cwd: PathBuf,
    managed_settings: PathBuf,
}
impl Fixture {
    fn layout(&self) -> Layout {
        Layout::new(&self.root, &self.cwd, &self.home).with_managed_settings(&self.managed_settings)
    }
}
fn build_fixture() -> Fixture {
    let tmp = tempfile::TempDir::new().unwrap();
    let base = tmp.path();
    let root = base.join("repo");
    let home = base.join("home");
    let cwd = root.join("service").join("api");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    write(
        &root.join("CLAUDE.md"),
        concat!(
            "# Root memory\n",
            "<!-- this html comment must be stripped -->\n",
            "Visible root line.\n",
            "@shared/base.md\n",
            "```\n",
            "@shared/should_not_import.md\n",
            "```\n",
            "Do not import `@shared/inline_ignore.md` either.\n",
        ),
    );
    write(&root.join("CLAUDE.local.md"), "Root local override.\n");
    write(
        &root.join("service").join("CLAUDE.md"),
        "# Service memory\n",
    );
    write(&cwd.join("CLAUDE.md"), "# API memory (most specific)\n");
    write(
        &cwd.join("v2").join("CLAUDE.md"),
        "# API v2 subtree memory\n",
    );
    write(&root.join("shared").join("base.md"), "[[BASE]]\n@a.md\n");
    write(&root.join("shared").join("a.md"), "[[A]]\n@b.md\n");
    write(&root.join("shared").join("b.md"), "[[B]]\n@c.md\n");
    write(&root.join("shared").join("c.md"), "[[C]]\n@d.md\n");
    write(&root.join("shared").join("d.md"), "[[D]]\n@e.md\n");
    write(&root.join("shared").join("e.md"), "[[E]]\n");
    write(
        &root.join(".claude").join("rules").join("general.md"),
        "# General rule (always on)\n",
    );
    write(
        &root.join(".claude").join("rules").join("rust.md"),
        "---\npaths:\n  - \"**/*.rs\"\n---\n# Rust-only rule\n",
    );
    write(
        &home.join(".claude").join("settings.json"),
        r#"{
  "model": "user-model",
  "instructions": "user-instructions",
  "permissions": { "allow": ["Bash(git status)", "Read(**)"] },
  "claudeMdExcludes": ["**/excluded/CLAUDE.md"]
}"#,
    );
    write(
        &root.join(".claude").join("settings.json"),
        r#"{
  "model": "project-model",
  "instructions": "project-instructions",
  "permissions": { "deny": ["Bash(git status)"] }
}"#,
    );
    write(
        &root.join(".claude").join("settings.local.json"),
        r#"{
  "model": "local-model",
  "instructions": "local-instructions",
  "permissions": { "ask": ["Bash(rm *)"] }
}"#,
    );
    let managed_settings = base.join("managed").join("settings.json");
    write(
        &managed_settings,
        r#"{
  "model": "managed-model",
  "instructions": "managed-instructions"
}"#,
    );
    write(
        &root.join("excluded").join("CLAUDE.md"),
        "# should be excluded\n",
    );
    write(
        &root.join(".claude").join("agents").join("reviewer.md"),
        concat!(
            "---\n",
            "name: reviewer\n",
            "description: reviews diffs\n",
            "tools: [Read, Write, Bash]\n",
            "disallowedTools: [Bash]\n",
            "skills: [lint]\n",
            "---\n",
            "You are a careful code reviewer.\n",
        ),
    );
    write(
        &root
            .join(".claude")
            .join("skills")
            .join("deploy")
            .join("SKILL.md"),
        concat!(
            "---\n",
            "name: deploy\n",
            "description: ship the service\n",
            "allowed-tools: [Bash, Read]\n",
            "disable-model-invocation: true\n",
            "effort: high\n",
            "---\n",
            "Steps to deploy.\n",
        ),
    );
    write(
        &home.join(".claude.json"),
        r#"{ "mcpServers": {
            "shared": { "command": "user-cmd", "args": ["--user"] },
            "user-only": { "command": "u" }
        } }"#,
    );
    write(
        &root.join(".mcp.json"),
        r#"{ "mcpServers": {
            "shared": { "command": "project-cmd" },
            "project-only": { "command": "p" }
        } }"#,
    );
    write(&root.join("AGENTS.md"), "# Cross-agent instructions\n");
    write(
        &root.join(".cursor").join("rules").join("style.mdc"),
        "cursor style rule\n",
    );
    Fixture {
        _tmp: tmp,
        root,
        home,
        cwd,
        managed_settings,
    }
}
#[test]
fn resolved_instruction_order_is_root_first_more_specific_last() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    let launch = cfg.memory.launch_order();
    let expected = vec![
        fx.root.join("CLAUDE.md"),
        fx.root.join("CLAUDE.local.md"),
        fx.root.join("service").join("CLAUDE.md"),
        fx.cwd.join("CLAUDE.md"),
    ];
    assert_eq!(launch, expected, "launch memory order");
    let lazy: Vec<PathBuf> = cfg
        .memory
        .lazy_entries()
        .iter()
        .map(|e| e.path.clone())
        .collect();
    assert_eq!(lazy, vec![fx.cwd.join("v2").join("CLAUDE.md")]);
    assert!(!cfg
        .memory
        .entries
        .iter()
        .any(|e| e.path.ends_with("excluded/CLAUDE.md")));
    let full = cfg.launch_instruction_order();
    assert_eq!(
        full.last().unwrap(),
        &fx.root.join(".claude").join("rules").join("general.md")
    );
    assert!(!full.iter().any(|p| p.ends_with("rules/rust.md")));
}
#[test]
fn html_comment_stripped_and_imports_inlined_with_depth_cutoff_and_fence_skip() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    let root_entry = cfg
        .memory
        .launch_entries()
        .into_iter()
        .find(|e| e.path == fx.root.join("CLAUDE.md"))
        .expect("root memory entry");
    let injected = &root_entry.injected;
    assert!(
        !injected.contains("must be stripped"),
        "html comment stripped"
    );
    assert!(injected.contains("Visible root line."));
    assert!(root_entry
        .imports
        .iter()
        .all(|i| !i.spec.contains("should_not_import")));
    assert!(root_entry
        .imports
        .iter()
        .all(|i| !i.spec.contains("inline_ignore")));
    for marker in ["[[BASE]]", "[[A]]", "[[B]]", "[[C]]"] {
        assert!(injected.contains(marker), "{marker} should be inlined");
    }
    assert!(
        !injected.contains("[[D]]"),
        "depth-5 import must be past the cutoff"
    );
    assert!(
        !injected.contains("[[E]]"),
        "an import beyond the cutoff is never reached"
    );
    let d_import = root_entry
        .imports
        .iter()
        .find(|i| i.spec == "d.md")
        .expect("d.md import recorded");
    assert!(!d_import.inlined, "d.md should be recorded but not inlined");
    assert_eq!(d_import.depth, claude_md::MAX_IMPORT_DEPTH + 1);
    assert!(root_entry.imports.iter().all(|i| i.spec != "e.md"));
    assert!(!cfg.memory.approval_required());
}
#[test]
fn external_import_is_flagged_for_approval() {
    let fx = build_fixture();
    let outside = fx.root.parent().unwrap().join("outside.md");
    write(&outside, "OUTSIDE_MARKER\n");
    write(
        &fx.root.join("CLAUDE.md"),
        &format!("Root.\n@{}\n", outside.display()),
    );
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    assert!(
        cfg.memory.approval_required(),
        "first external import must be flagged for approval"
    );
    let root_entry = cfg
        .memory
        .launch_entries()
        .into_iter()
        .find(|e| e.path == fx.root.join("CLAUDE.md"))
        .unwrap();
    let ext = root_entry
        .imports
        .iter()
        .find(|i| i.external)
        .expect("external import present");
    assert!(ext.approval_required);
    assert!(
        ext.inlined,
        "external file still inlines once approved-flagged"
    );
}
#[test]
fn permission_merge_deny_wins() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    assert_eq!(cfg.settings.decide("Bash(git status)"), Decision::Deny);
    assert_eq!(cfg.settings.decide("Read(src/main.rs)"), Decision::Allow);
    assert_eq!(cfg.settings.decide("Bash(rm -rf tmp)"), Decision::Ask);
    assert_eq!(cfg.settings.decide("Write(x)"), Decision::Undecided);
}
#[test]
fn scalar_precedence_and_instruction_precedence_are_separate() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    assert_eq!(
        cfg.settings.values.get("model").and_then(|v| v.as_str()),
        Some("managed-model")
    );
    let order: Vec<Scope> = cfg
        .settings
        .instruction_layers
        .iter()
        .map(|(s, _)| *s)
        .collect();
    assert_eq!(
        order,
        vec![Scope::Local, Scope::Project, Scope::User, Scope::Managed]
    );
    assert_eq!(
        cfg.settings.effective_instructions(),
        Some("managed-instructions")
    );
}
#[test]
fn rule_path_glob_gating() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    let rust_rule = cfg
        .rules
        .iter()
        .find(|r| r.path.ends_with("rules/rust.md"))
        .expect("rust rule present");
    assert!(rust_rule.scoped);
    assert!(!rust_rule.loads_at_launch());
    assert!(rust_rule.applies_to(&fx.root.join("service/api/main.rs"), &fx.root));
    assert!(!rust_rule.applies_to(&fx.root.join("README.md"), &fx.root));
    let general = cfg
        .rules
        .iter()
        .find(|r| r.path.ends_with("rules/general.md"))
        .expect("general rule present");
    assert!(!general.scoped);
    assert!(general.loads_at_launch());
    assert!(general.applies_to(&fx.root.join("anything.txt"), &fx.root));
}
#[test]
fn agent_frontmatter_parse_with_disallowed_before_allow() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    let agent = cfg
        .agents
        .iter()
        .find(|a| a.name == "reviewer")
        .expect("reviewer agent parsed");
    assert_eq!(agent.description.as_deref(), Some("reviews diffs"));
    assert_eq!(agent.tools, vec!["Read", "Write", "Bash"]);
    assert_eq!(agent.disallowed_tools, vec!["Bash"]);
    assert_eq!(agent.effective_tools(), vec!["Read", "Write"]);
    assert!(agent.inherits_model(), "model defaults to inherit");
    assert_eq!(agent.skills, vec!["lint"]);
    assert!(agent.body.contains("careful code reviewer"));
}
#[test]
fn skill_frontmatter_parse() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    let skill = cfg
        .skills
        .iter()
        .find(|s| s.name == "deploy")
        .expect("deploy skill parsed");
    assert_eq!(skill.allowed_tools, vec!["Bash", "Read"]);
    assert!(!skill.model_invocable(), "disable-model-invocation honored");
    assert!(skill.user_invocable, "user-invocable defaults true");
    assert_eq!(skill.effort.as_deref(), Some("high"));
}
#[test]
fn mcp_whole_entry_precedence() {
    let fx = build_fixture();
    let cfg = CompatConfig::load(&fx.layout(), None).unwrap();
    let shared = cfg.mcp.server("shared").expect("shared server");
    assert_eq!(shared.scope, mcp::McpScope::Project);
    assert_eq!(shared.command(), Some("project-cmd"));
    assert!(shared.entry.get("args").is_none());
    assert!(cfg.mcp.server("user-only").is_some());
    assert!(cfg.mcp.server("project-only").is_some());
    assert!(cfg
        .mcp
        .agents_md
        .as_deref()
        .unwrap()
        .contains("Cross-agent"));
    assert_eq!(cfg.mcp.cursor_rules.len(), 1);
    assert!(cfg.mcp.cursor_rules[0].body.contains("cursor style rule"));
}
#[test]
fn managed_mcp_overrides_project() {
    let fx = build_fixture();
    let managed_mcp = fx.root.parent().unwrap().join("managed_mcp.json");
    write(
        &managed_mcp,
        r#"{ "mcpServers": { "shared": { "command": "managed-cmd" } } }"#,
    );
    let layout = fx.layout().with_managed_mcp(&managed_mcp);
    let cfg = CompatConfig::load(&layout, None).unwrap();
    let shared = cfg.mcp.server("shared").unwrap();
    assert_eq!(shared.scope, mcp::McpScope::Managed);
    assert_eq!(shared.command(), Some("managed-cmd"));
}
#[test]
fn cli_scalar_scope_beats_local_but_not_managed() {
    let fx = build_fixture();
    let cli = settings::parse_value(serde_json::json!({ "model": "cli-model" }));
    let layout = Layout::new(&fx.root, &fx.cwd, &fx.home);
    let cfg = CompatConfig::load(&layout, Some(cli)).unwrap();
    assert_eq!(
        cfg.settings.values.get("model").and_then(|v| v.as_str()),
        Some("cli-model")
    );
}
