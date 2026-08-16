//! Receipt scanner: flag a parallel sum where a critical-path number belongs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlagSeverity {
    P1,
    P2,
    Note,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReceiptFlag {
    pub path: String,
    pub severity: FlagSeverity,
    pub kind: String,
    pub detail: String,
}

fn json_f64(v: &Value, keys: &[&str]) -> Option<f64> {
    let mut cur = v;
    for k in keys {
        cur = cur.get(*k)?;
    }
    cur.as_f64()
        .or_else(|| cur.as_u64().map(|u| u as f64))
        .or_else(|| cur.as_i64().map(|i| i as f64))
}

fn json_bool(v: &Value, keys: &[&str]) -> Option<bool> {
    let mut cur = v;
    for k in keys {
        cur = cur.get(*k)?;
    }
    cur.as_bool()
}

fn json_str<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    let mut cur = v;
    for k in keys {
        cur = cur.get(*k)?;
    }
    cur.as_str()
}

pub fn flag_receipt(path: &str, root: &Value) -> Vec<ReceiptFlag> {
    let mut flags = Vec::new();
    let body = json_f64(root, &["body_ns"])
        .or_else(|| json_f64(root, &["token_ns_ledger", "body_ns"]))
        .unwrap_or(0.0);
    let verify = json_f64(root, &["verify_ns"])
        .or_else(|| json_f64(root, &["token_ns_ledger", "verify_ns"]))
        .or_else(|| json_f64(root, &["chunk_verification", "verify_ns"]))
        .unwrap_or(0.0);
    if body > 0.0 && verify > body {
        let labeled_parallel = json_str(root, &["verification_per_token", "note"])
            .map(|n| n.contains("parallel") || n.contains("SUM"))
            .unwrap_or(false)
            || json_str(root, &["verification_per_token", "note"])
                .map(|n| n.contains("not wall"))
                .unwrap_or(false);
        flags.push(ReceiptFlag {
            path: path.to_owned(),
            severity: if labeled_parallel {
                FlagSeverity::Note
            } else {
                FlagSeverity::P1
            },
            kind: "parallel_sum_as_verify_ns".into(),
            detail: format!(
                "verify_ns={:.0} > body_ns={:.0} ({:.0}% of body). {}",
                verify,
                body,
                verify * 100.0 / body,
                if labeled_parallel {
                    "Receipt labels this as a parallel sum — correct."
                } else {
                    "Presented without a parallel-sum label. This is not token latency."
                }
            ),
        });
    }

    let stages = root
        .get("stages")
        .or_else(|| root.pointer("/token_ns_ledger/stages"))
        .and_then(|s| s.as_array());
    if let Some(stages) = stages {
        for row in stages {
            let name = json_str(row, &["name"]).unwrap_or("");
            let ns = json_f64(row, &["ns_per_token"])
                .or_else(|| json_f64(row, &["ns"]))
                .unwrap_or(0.0);
            let pct = json_f64(row, &["pct_body"]).unwrap_or(0.0);
            if name.starts_with("reader.") && ns > 0.0 && body > 0.0 {
                let pct_use = if pct > 0.0 { pct } else { ns * 100.0 / body };
                flags.push(ReceiptFlag {
                    path: path.to_owned(),
                    severity: if ns > body {
                        FlagSeverity::P1
                    } else {
                        FlagSeverity::Note
                    },
                    kind: "parallel_sum_as_stage_ns".into(),
                    detail: format!(
                        "{name} ns={ns:.0} is {pct_use:.0}% of body_ns={body:.0}. This is a parallel-thread sum, not critical-path latency."
                    ),
                });
            }
        }
    }

    if json_bool(root, &["claim_boundary", "not_full_token_runtime"]) == Some(true) {
        if let Some(result) = json_f64(root, &["result_ns_per_token"]) {
            flags.push(ReceiptFlag {
                path: path.to_owned(),
                severity: FlagSeverity::P1,
                kind: "isolated_product_as_result_ns_per_token".into(),
                detail: format!(
                    "result_ns_per_token={result:.0} but claim_boundary.not_full_token_runtime=true. This number is not a token."
                ),
            });
        }
    }

    if path.ends_with("dsv-cb-collapse.json") {
        if let Some(arr) = root.pointer("/spread_body_ns/A_baseline").and_then(|v| v.as_array()) {
            let vals: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
            if vals.len() >= 3 && vals[0] > 2.0 * vals[1] {
                flags.push(ReceiptFlag {
                    path: path.to_owned(),
                    severity: FlagSeverity::P1,
                    kind: "cold_contaminated_median".into(),
                    detail: format!(
                        "A bodies [{:.0}, {:.0}, {:.0}] ns. First A is a cold 12.9 s run; the published median is not a critical-path number vs the 1038 ms host-wall authority.",
                        vals[0], vals[1], vals[2]
                    ),
                });
            }
        }
    }

    if path.contains("HOST_WALL_BASELINE") {
        if let Some(note) = json_str(root, &["verification_per_token", "note"]) {
            if note.contains("parallel") {
                flags.push(ReceiptFlag {
                    path: path.to_owned(),
                    severity: FlagSeverity::Note,
                    kind: "parallel_sum_correctly_labeled".into(),
                    detail: "Host-wall baseline correctly states verify_ns/path_resolve are parallel sums, not wall.".into(),
                });
            }
        }
    }

    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_reader_overcount() {
        let v = serde_json::json!({
            "body_ns": 1000,
            "verify_ns": 2500,
            "stages": [
                {"name": "reader.path_resolve", "ns": 1300, "ns_per_token": 1300, "pct_body": 130.0}
            ]
        });
        let flags = flag_receipt("ledger.json", &v);
        assert!(flags.iter().any(|f| f.kind == "parallel_sum_as_stage_ns"));
        assert!(flags.iter().any(|f| f.kind == "parallel_sum_as_verify_ns"));
        assert!(flags.iter().any(|f| f.severity == FlagSeverity::P1));
    }

    #[test]
    fn flags_isolated_organ_product() {
        let v = serde_json::json!({
            "claim_boundary": {"not_full_token_runtime": true},
            "result_ns_per_token": 192000000
        });
        let flags = flag_receipt("q80-decode-kernels.json", &v);
        assert_eq!(flags[0].kind, "isolated_product_as_result_ns_per_token");
    }
}
