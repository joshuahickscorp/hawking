//! End-to-end proofs for the sandboxed program runtime.
//!
//! These are deterministic and model-free: every test runs a program (an AST)
//! over fixture handles and asserts on the structured result, the enforced
//! limits, write-separation, the absence of ambient authority, and byte-level
//! determinism. No model, no network, no real browser, no filesystem.

use std::collections::BTreeMap;

use hide_program_runtime::{
    map_of, run, BinOp, Citation, DenyAllHost, Expr, HandleError, HandleGrants, HandleName,
    HostHandles, JoinKind, Lambda, LimitKind, Limits, Operator, Order, Program, RetryPolicy,
    RuntimeError, Value, WriteKind,
};

// --- fixtures ----------------------------------------------------------------

/// A deterministic host that serves canned, cited rows for two read handles and
/// refuses everything else. Stands in for the real read tools.
struct FixtureHost;

impl FixtureHost {
    fn hit(path: &str, score: i64, owner: &str) -> Value {
        map_of([
            ("path", Value::from(path)),
            ("score", Value::from(score)),
            ("owner", Value::from(owner)),
        ])
        .with_merged_citations(&[Citation::new("search.text")
            .with_locator(path)
            .with_digest("deadbeef")])
    }

    fn owner_row(owner: &str, commits: i64) -> Value {
        map_of([
            ("owner", Value::from(owner)),
            ("commits", Value::from(commits)),
        ])
        .with_merged_citations(&[Citation::new("git.log").with_locator(owner)])
    }
}

impl HostHandles for FixtureHost {
    fn call(&self, handle: HandleName, _args: &Value, _attempt: u32) -> Result<Value, HandleError> {
        match handle {
            HandleName::SearchText => Ok(Value::List(vec![
                Self::hit("core/a.rs", 3, "ana"),
                Self::hit("core/b.rs", 9, "bo"),
                Self::hit("core/c.rs", 7, "ana"),
                Self::hit("core/d.rs", 1, "cy"),
            ])),
            HandleName::GitLog => Ok(Value::List(vec![
                Self::owner_row("ana", 12),
                Self::owner_row("bo", 4),
            ])),
            other => Err(HandleError::new(other.as_str(), "not served by fixture")),
        }
    }
}

fn grants() -> HandleGrants {
    HandleGrants::of([HandleName::SearchText, HandleName::GitLog])
}

// A field access helper for assertions.
fn field<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get_path(&[key.to_string()])
}

// --- headline: map -> filter -> rank -> join with preserved citations --------

/// Build the analysis program used by several tests: search, project rows
/// (preserving citations), filter by score, rank desc, and inner-join with the
/// owners from git.log.
fn analysis_program() -> Expr {
    let hits = Expr::handle(HandleName::SearchText, Expr::lit("needle"));

    // parallel_map-shaped projection wrapped in citation_preservation so the
    // projected rows keep their source citations even though the lambda drops
    // the reserved field.
    let projected = Expr::op(Operator::CitationPreservation {
        input: Box::new(hits),
        func: Lambda::new(
            "h",
            Expr::map_lit([
                ("path", Expr::field(Expr::var("h"), ["path"])),
                ("score", Expr::field(Expr::var("h"), ["score"])),
                ("owner", Expr::field(Expr::var("h"), ["owner"])),
            ]),
        ),
    });

    let filtered = Expr::op(Operator::Filter {
        input: Box::new(projected),
        pred: Lambda::new(
            "r",
            Expr::bin(BinOp::Ge, Expr::field(Expr::var("r"), ["score"]), Expr::lit(5i64)),
        ),
    });

    let ranked = Expr::op(Operator::Rank {
        input: Box::new(filtered),
        key: Lambda::new("r", Expr::field(Expr::var("r"), ["score"])),
        order: Order::Desc,
        limit: None,
    });

    let owners = Expr::handle(HandleName::GitLog, Expr::lit(Value::Null));

    let joined = Expr::op(Operator::Join {
        left: Box::new(ranked),
        right: Box::new(owners),
        left_key: Lambda::new("r", Expr::field(Expr::var("r"), ["owner"])),
        right_key: Lambda::new("o", Expr::field(Expr::var("o"), ["owner"])),
        kind: JoinKind::Inner,
    });

    Expr::map_lit([("results", joined)])
}

#[test]
fn map_filter_rank_join_preserves_citations() {
    let prog = Program::new(analysis_program());
    let out = run(&prog, &FixtureHost, &grants(), Limits::strict()).unwrap();

    let results = field(&out.value, "results").unwrap().as_list().unwrap();

    // score>=5 keeps b.rs(9) and c.rs(7); both owners (bo, ana) are in git.log,
    // so the inner join yields exactly two rows, ranked by score desc.
    assert_eq!(results.len(), 2);

    let first = &results[0];
    assert_eq!(field(field(first, "left").unwrap(), "path"), Some(&Value::from("core/b.rs")));
    assert_eq!(field(field(first, "left").unwrap(), "score"), Some(&Value::from(9i64)));
    assert_eq!(field(field(first, "right").unwrap(), "commits"), Some(&Value::from(4i64)));

    // Citations survived projection AND join: each result carries evidence from
    // both search.text (the hit) and git.log (the owner).
    for row in results {
        let sources: Vec<String> = row.citations().into_iter().map(|c| c.source).collect();
        assert!(sources.contains(&"search.text".to_string()), "missing search cite: {sources:?}");
        assert!(sources.contains(&"git.log".to_string()), "missing git cite: {sources:?}");
    }

    // The second ranked row is c.rs(7), owner ana.
    let second = &results[1];
    assert_eq!(field(field(second, "left").unwrap(), "path"), Some(&Value::from("core/c.rs")));
    assert_eq!(field(field(second, "right").unwrap(), "commits"), Some(&Value::from(12i64)));
}

// --- determinism: byte-identical output across two runs ----------------------

#[test]
fn same_program_same_input_is_byte_identical() {
    // Add a seeded sample so determinism is a non-trivial claim.
    let sampled = Expr::op(Operator::Sample {
        input: Box::new(Expr::handle(HandleName::SearchText, Expr::lit("needle"))),
        k: 2,
    });
    let root = Expr::map_lit([("analysis", analysis_program()), ("sampled", sampled)]);
    let prog = Program::new(root).with_seed(1234);

    let a = run(&prog, &FixtureHost, &grants(), Limits::strict()).unwrap();
    let b = run(&prog, &FixtureHost, &grants(), Limits::strict()).unwrap();

    let ba = serde_json::to_vec(&a).unwrap();
    let bb = serde_json::to_vec(&b).unwrap();
    assert_eq!(ba, bb, "two runs must be byte-identical");
    // And the deterministic usage snapshot matches too.
    assert_eq!(a.usage, b.usage);
}

// --- write separation: prepare, never execute --------------------------------

#[test]
fn program_that_wants_to_edit_returns_a_proposal_and_writes_nothing() {
    // A program with NO grants and a deny-all host: it cannot read the world at
    // all. It can still prepare a write, proving proposals need no authority and
    // execute nowhere.
    let spec = Expr::map_lit([
        ("kind", Expr::lit("edit")),
        ("summary", Expr::lit("rename Foo to Bar in core/a.rs")),
        (
            "payload",
            Expr::map_lit([
                ("path", Expr::lit("core/a.rs")),
                ("diff", Expr::lit("@@ -1 +1 @@\n-Foo\n+Bar\n")),
            ]),
        ),
        (
            "citations",
            Expr::lit(Value::List(vec![map_of([("source", Value::from("file.read"))])])),
        ),
    ]);
    let root = Expr::map_lit([("staged", Expr::propose_write(spec))]);

    let prog = Program::new(root);
    let out = run(&prog, &DenyAllHost, &HandleGrants::none(), Limits::strict()).unwrap();

    // Exactly one proposal, correctly typed, and it carries provenance.
    assert_eq!(out.proposals.len(), 1);
    let p = &out.proposals[0];
    assert_eq!(p.kind, WriteKind::Edit);
    assert_eq!(p.summary, "rename Foo to Bar in core/a.rs");
    assert_eq!(field(&p.payload, "path"), Some(&Value::from("core/a.rs")));
    assert_eq!(p.citations.len(), 1);
    assert_eq!(p.citations[0].source, "file.read");

    // The runtime executed nothing: no artifacts, and it never called a handle
    // (there is no write handle to call, and no read grant either).
    assert!(out.artifacts.is_empty());
    assert_eq!(out.usage.tool_calls, 0);

    // The returned value only references the proposal by id; it is not a write.
    let staged = field(&out.value, "staged").unwrap();
    assert_eq!(field(staged, "@write_proposal"), Some(&Value::from("wp-0")));
}

// --- no ambient authority: impossible by construction ------------------------

#[test]
fn there_is_no_fs_network_or_subprocess_handle_to_name() {
    // The handle vocabulary is closed to ten read-oriented names. None of the
    // dangerous capabilities can even be spelled.
    for forbidden in [
        "fs.write",
        "file.write",
        "process.spawn",
        "shell.exec",
        "net.connect",
        "http.get",
        "env.read",
        "secret.read",
        "credentials.get",
    ] {
        assert!(HandleName::from_str(forbidden).is_none(), "{forbidden} must not exist");
    }
    // Every handle that DOES exist is read-oriented.
    assert!(HandleName::ALL.iter().all(HandleName::is_read_oriented));
    assert_eq!(HandleName::ALL.len(), 10);
}

#[test]
fn calling_an_ungranted_handle_is_denied_before_the_host_is_touched() {
    // Grant only search.text; the program reaches for file.read.
    let prog = Program::new(Expr::handle(HandleName::FileRead, Expr::lit("/etc/passwd")));
    let err = run(
        &prog,
        &FixtureHost,
        &HandleGrants::of([HandleName::SearchText]),
        Limits::strict(),
    )
    .unwrap_err();
    assert_eq!(err, RuntimeError::HandleNotGranted("file.read".into()));
}

// --- limits: each dimension trips a typed error ------------------------------

/// Start from unbounded and tighten a single dimension.
fn only(mut mutate: impl FnMut(&mut Limits)) -> Limits {
    let mut l = Limits::unbounded();
    mutate(&mut l);
    l
}

#[test]
fn instruction_limit_trips() {
    // Add(Add(1,2),3) evaluates more than 3 nodes.
    let e = Expr::bin(
        BinOp::Add,
        Expr::bin(BinOp::Add, Expr::lit(1i64), Expr::lit(2i64)),
        Expr::lit(3i64),
    );
    let err = run(&Program::new(e), &DenyAllHost, &HandleGrants::none(), only(|l| l.instructions = 3))
        .unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::Instruction));
}

#[test]
fn wall_time_limit_trips() {
    let e = Expr::handle(HandleName::SearchText, Expr::lit("q"));
    let limits = only(|l| {
        l.wall_time_ms = 3;
        l.handle_latency_ms = 10;
    });
    let err = run(&Program::new(e), &FixtureHost, &grants(), limits).unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::WallTime));
}

#[test]
fn output_bytes_limit_trips() {
    let big: Vec<Value> = (0..200).map(Value::Int).collect();
    let e = Expr::lit(Value::List(big));
    let err = run(&Program::new(e), &DenyAllHost, &HandleGrants::none(), only(|l| l.output_bytes = 16))
        .unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::OutputBytes));
}

#[test]
fn tool_call_limit_trips() {
    // Two handle calls under a budget of one.
    let e = Expr::list([
        Expr::handle(HandleName::SearchText, Expr::lit("a")),
        Expr::handle(HandleName::SearchText, Expr::lit("b")),
    ]);
    let err = run(&Program::new(e), &FixtureHost, &grants(), only(|l| l.tool_calls = 1)).unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::ToolCall));
}

#[test]
fn memory_limit_trips() {
    // The fixture returns rows far larger than a 16-byte peak budget.
    let e = Expr::handle(HandleName::SearchText, Expr::lit("q"));
    let err = run(&Program::new(e), &FixtureHost, &grants(), only(|l| l.memory_bytes = 16)).unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::Memory));
}

#[test]
fn concurrency_limit_trips() {
    let e = Expr::op(Operator::ParallelMap {
        input: Box::new(Expr::lit(Value::List(vec![Value::Int(1)]))),
        func: Lambda::new("x", Expr::var("x")),
        concurrency: Some(8),
    });
    let err = run(&Program::new(e), &DenyAllHost, &HandleGrants::none(), only(|l| l.concurrency = 4))
        .unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::Concurrency));
}

#[test]
fn artifact_byte_limit_trips() {
    let big: Vec<Value> = (0..50).map(Value::Int).collect();
    let e = Expr::op(Operator::SpillToArtifact {
        input: Box::new(Expr::lit(Value::List(big))),
        name: "scratch".into(),
    });
    let err = run(&Program::new(e), &DenyAllHost, &HandleGrants::none(), only(|l| l.artifact_bytes = 8))
        .unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::ArtifactByte));
}

#[test]
fn recursion_limit_trips() {
    // Nest Not() well past the depth budget.
    let mut e = Expr::lit(true);
    for _ in 0..40 {
        e = Expr::not(e);
    }
    let err = run(&Program::new(e), &DenyAllHost, &HandleGrants::none(), only(|l| l.recursion_depth = 5))
        .unwrap_err();
    assert_eq!(err.limit_kind(), Some(LimitKind::Recursion));
}

// --- spill + artifact success path -------------------------------------------

#[test]
fn spill_to_artifact_records_content_out_of_band() {
    let data = Value::List((0..10).map(Value::Int).collect());
    let e = Expr::op(Operator::SpillToArtifact {
        input: Box::new(Expr::lit(data.clone())),
        name: "rows".into(),
    });
    let out = run(&Program::new(e), &DenyAllHost, &HandleGrants::none(), Limits::strict()).unwrap();

    assert_eq!(out.artifacts.len(), 1);
    let art = &out.artifacts[0];
    assert_eq!(art.name, "rows");
    assert_eq!(art.content, data);
    // The returned value is a small reference, not the payload.
    assert_eq!(field(&out.value, "@artifact"), Some(&Value::from("artifact-0")));
    assert_eq!(field(&out.value, "name"), Some(&Value::from("rows")));
}

// --- group + retry exhaustion ------------------------------------------------

#[test]
fn group_buckets_by_key_deterministically() {
    let e = Expr::op(Operator::Group {
        input: Box::new(Expr::handle(HandleName::SearchText, Expr::lit("q"))),
        key: Lambda::new("h", Expr::field(Expr::var("h"), ["owner"])),
    });
    let out = run(&Program::new(e), &FixtureHost, &grants(), Limits::strict()).unwrap();
    let groups = out.value.as_list().unwrap();
    // Owners ana, bo, cy sorted by canonical key.
    let keys: Vec<&str> = groups
        .iter()
        .map(|g| field(g, "key").unwrap().as_str().unwrap())
        .collect();
    assert_eq!(keys, vec!["ana", "bo", "cy"]);
    // ana owns two hits.
    let ana_items = field(&groups[0], "items").unwrap().as_list().unwrap();
    assert_eq!(ana_items.len(), 2);
}

#[test]
fn retry_exhaustion_surfaces_the_handle_error() {
    // Always-flaky handle; retry runs out of attempts and returns the error.
    struct Flaky;
    impl HostHandles for Flaky {
        fn call(&self, _h: HandleName, _a: &Value, _t: u32) -> Result<Value, HandleError> {
            Err(HandleError::retryable("git.log", "always transient"))
        }
    }
    let e = Expr::op(Operator::RetryWithPolicy {
        body: Box::new(Expr::handle(HandleName::GitLog, Expr::lit(Value::Null))),
        policy: RetryPolicy::new(3, 1),
    });
    let err = run(
        &Program::new(e),
        &Flaky,
        &HandleGrants::of([HandleName::GitLog]),
        Limits::strict(),
    )
    .unwrap_err();
    match err {
        RuntimeError::Handle(h) => assert_eq!(h.message, "always transient"),
        other => panic!("expected a handle error, got {other:?}"),
    }
}

// --- a small program authored as JSON round-trips and runs -------------------

#[test]
fn a_program_authored_as_json_deserializes_and_runs() {
    // Programs are data: this one is written as JSON, parsed, and executed.
    let json = r#"
    {
      "root": {
        "expr": "op",
        "operator": {
          "op": "filter",
          "input": { "expr": "lit", "value": [ {"n": 1}, {"n": 5}, {"n": 9} ] },
          "pred": {
            "param": "r",
            "body": {
              "expr": "bin_op",
              "op": "ge",
              "lhs": { "expr": "field", "base": { "expr": "var", "name": "r" }, "path": ["n"] },
              "rhs": { "expr": "lit", "value": 5 }
            }
          }
        }
      },
      "seed": 0,
      "clock_start_ms": 0
    }"#;

    let prog: Program = serde_json::from_str(json).unwrap();
    let out = run(&prog, &DenyAllHost, &HandleGrants::none(), Limits::strict()).unwrap();
    let kept = out.value.as_list().unwrap();
    assert_eq!(kept.len(), 2);
    assert_eq!(field(&kept[0], "n"), Some(&Value::from(5i64)));
    assert_eq!(field(&kept[1], "n"), Some(&Value::from(9i64)));

    // Silence the unused import lints when only some helpers are used.
    let _ = BTreeMap::<String, Value>::new();
}
