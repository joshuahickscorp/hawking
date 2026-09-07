//! STATIC_ONLY host/shader ABI preflight.
//!
//! Port of `tools/future/static_kernel_verify.py`. A finding of no static ERROR
//! is not a tps number. This exists to stop wasting a protected GPU window on a
//! defect that was detectable from `.metal` sources and their Rust hosts.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const SCHEMA: &str = "hawking.future.static_kernel_verify.v1";
pub const VERSION: u32 = 1;
pub const APPLE_MAX_THREADS_PER_THREADGROUP: i64 = 1024;
pub const APPLE_SIMDGROUP_WIDTH: i64 = 32;

pub const KIND_DEVICE: &str = "device";
pub const KIND_CONSTANT_U32: &str = "constant_u32";
pub const KIND_CONSTANT_F32: &str = "constant_f32";
pub const KIND_CONSTANT_STRUCT: &str = "constant_struct";
pub const KIND_CONSTANT_BYTES: &str = "constant_bytes";
pub const KIND_THREADGROUP: &str = "threadgroup";
pub const KIND_UNKNOWN: &str = "unknown";

const SHADER_DIR: &str = "crates/hawking-core/shaders";
const HOST_ROOT: &str = "crates/hawking-core";
const PRODUCTION_HOST_PREFIX: &str = "crates/hawking-core/src/";

const STATIC_CORRECTNESS_IS_NOT_SPEED: &str = concat!(
    "Static correctness does NOT prove speed and does NOT substitute for a ",
    "protected measurement. A finding of no static ERROR is not a tps number, ",
    "not a token_ns, not a complete-token, and not PROTECTED_ABSOLUTE. This ",
    "tool exists to stop wasting a protected GPU window on a defect that was ",
    "detectable from the .metal sources and their Rust hosts. It produces ",
    "neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. Everything here is ",
    "STATIC_ONLY with bench state UNKNOWN."
);

static KERNEL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bkernel\s+void\s+(\w+)\s*\(").unwrap());
static STRUCT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bstruct\s+(\w+)\s*\{").unwrap());
static ATTR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[\[([^\]]+)\]\]").unwrap());
static BUF_IDX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"buffer\((\d+)\)").unwrap());
static TG_IDX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"threadgroup\((\d+)\)").unwrap());
static PARAM_NAME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([A-Za-z_]\w*)\s*$").unwrap());
static FIELD_NAME_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"([A-Za-z_]\w*)\s*(?:\[(\d+)\])?\s*$").unwrap());
static QWEN_K: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*QWEN_UNIFORM_Q4_MATMUL_K\(\s*(\d+)\s*\)").unwrap());
static QWEN_RK: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^\s*QWEN_UNIFORM_Q4_MATMUL_RK\(\s*(\d+)\s*,\s*(\d+)\s*\)").unwrap()
});
static QWEN_BP: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*QWEN_BINARY_PLANES\(\s*(\d+)\s*\)").unwrap());
static DISPATCH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\.dispatch_threads(?:_timed|_in_concurrent_group|_pair_in_one_encoder)?\s*\(")
        .unwrap()
});
static CFG_FEATURE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"#\[cfg\((?:feature\s*=\s*"([^"]+)"|all\([^]]*feature\s*=\s*"([^"]+)"[^]]*)\)\]"#)
        .unwrap()
});
static CONST_STR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?:pub(?:\([^)]+\))?\s+)?(?:const|static)\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:'static\s+)?str\s*=\s*"([A-Za-z_][A-Za-z0-9_]*)""#,
    )
    .unwrap()
});
static CONST_U32_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"const\s+([A-Z][A-Z0-9_]*)\s*:\s*u32\s*=\s*(\d+)").unwrap());
static ARGLAYOUT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"KernelArgBuffer::new\s*\([^,]+,\s*&\[([^\]]+)\]").unwrap());
static REPR_C_STRUCT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"#\[repr\(C[^\]]*\)\](?:\s*#\[[^\]]+\])*\s*(?:pub(?:\([^)]+\))?\s+)?struct\s+(\w+)\s*\{",
    )
    .unwrap()
});
static INCLUDE_STR_SHADER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"include_str!\s*\(\s*(?:concat!\([^)]*\))?\s*[^)]*shaders/([^"\)]+\.metal)"#)
        .unwrap()
});
static SHADERS_METAL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"shaders/([A-Za-z0-9_]+\.metal)").unwrap());
static PICK_FN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"pub fn (\w+)\s*\(\s*\)\s*->\s*&'static str\s*\{[^}]*pick\(\s*([A-Z0-9_]+)\s*,\s*([A-Z0-9_]+)",
    )
    .unwrap()
});
static FN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)").unwrap()
});
static STRING_LIT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#""([A-Za-z_][A-Za-z0-9_]*)""#).unwrap());
static METHOD_BIND_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?P<recv>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?P<meth>[A-Za-z0-9_]*set_u32|[A-Za-z0-9_]*set_f32|set_buffer|set_bytes|set_threadgroup_memory_length)\s*\(\s*(?P<idx>[^,\)]+)",
    )
    .unwrap()
});
static RUST_FIELD_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)(?:pub(?:\([^)]+\))?\s+)?([A-Za-z_]\w*)\s*:\s*(.+)$").unwrap());
static AS_U32_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+as\s+u32$").unwrap());
static AS_U_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+as\s+u(?:32|64)$").unwrap());
static INT_LIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\d+)(?:u32|u64|usize)?$").unwrap());
static WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static KERNEL_NAME_LIT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"^"([A-Za-z_][A-Za-z0-9_]*)"$"#).unwrap());
static DECODE_FAMILY_CALL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"decode_family::(\w+)\s*\(\s*\)").unwrap());
static UPPER_IDENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Z][A-Z0-9_]*$").unwrap());
static IDENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").unwrap());
static DIGITS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+$").unwrap());
static CLOSURE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\|[^|]*\|\s*(\{)?").unwrap());
static HELPER_CALL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([A-Za-z_]\w+)\s*\(").unwrap());
static ANY_CALL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*\s*\(").unwrap());
static ARGLAYOUT_NAME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"ArgLayout::(\w+)").unwrap());
static SHADER_CONST_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(SHADER_[A-Z0-9_]+)\s*:\s*&str\s*=\s*include_str!\s*\(\s*"\.\./\.\./shaders/([^"]+)""#,
    )
    .unwrap()
});
static ALL_SHADER_FN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"fn all_shader_sources\s*\(\s*\)[^{]*\{").unwrap());
static CFG_MOD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"#\[cfg\((?:feature\s*=\s*"([^"]+)"|all\([^]]*feature\s*=\s*"([^"]+)"[^]]*)\)\]\s*(?:pub(?:\([^)]+\))?\s+)?(?:mod|use)\s+([A-Za-z_][A-Za-z0-9_]*)"#,
    )
    .unwrap()
});
static FAMILY_ARRAY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?s)(?:FAMILY_KERNELS|Q80_GRAPH_KERNELS|Q80_TILE_KERNELS|Q80_GRAPH_SIMD_KERNELS|QWEN38_GRAPH_KERNELS|DSV4F_GRAPH_KERNELS)\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\]",
    )
    .unwrap()
});
static FAMILY_IDENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([A-Z][A-Z0-9_]*)").unwrap());
static METAL_ARRAY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(.+?)\s*\[(\d+)\]$").unwrap());
static RUST_ARRAY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\[(.+);\s*(\d+)\]$").unwrap());

const ARGBUF_RECV: &[&str] = &["ab", "argbuf", "args", "layout", "arg_buf", "arg_buffer"];

#[derive(Clone, Debug)]
struct ShaderParam {
    name: String,
    type_s: String,
    space: String,
    index: Option<i64>,
    kind: String,
}

#[derive(Clone, Debug)]
struct MetalKernel {
    name: String,
    path: String,
    line: usize,
    buffer_indices: BTreeMap<i64, ShaderParam>,
    threadgroup_indices: BTreeMap<i64, ShaderParam>,
}

impl MetalKernel {
    fn buffer_set(&self) -> HashSet<i64> {
        self.buffer_indices.keys().copied().collect()
    }
}

#[derive(Clone, Debug)]
struct StructField {
    name: String,
    type_s: String,
}

#[derive(Clone, Debug)]
struct StructDef {
    name: String,
    path: String,
    line: usize,
    fields: Vec<StructField>,
    lang: String,
    size: Option<i64>,
    align: Option<i64>,
    layout: Option<Vec<LayoutRow>>,
}

#[derive(Clone, Debug)]
struct LayoutRow {
    name: String,
    type_s: String,
    offset: i64,
    size: i64,
}

#[derive(Clone, Debug)]
struct HostBind {
    kind: String,
    index: Option<i64>,
    line: usize,
    hint: String,
}

#[derive(Clone, Debug)]
struct HostDispatch {
    path: String,
    line: usize,
    kernel_expr: String,
    resolved: Vec<String>,
    resolve_status: String,
    grid_raw: String,
    tg_raw: String,
    grid: (Option<i64>, Option<i64>, Option<i64>),
    tg: (Option<i64>, Option<i64>, Option<i64>),
    binds: Vec<HostBind>,
    binds_unverifiable: bool,
    feature_cfg: Option<String>,
    argbuf_layouts: BTreeMap<i64, Vec<String>>,
}

#[derive(Clone, Debug)]
pub struct Finding {
    pub severity: String,
    pub check: String,
    pub message: String,
    pub host: Option<String>,
    pub shader: Option<String>,
    pub kernel: Option<String>,
    pub extra: Map<String, Value>,
}

impl Finding {
    fn new(
        severity: &str,
        check: &str,
        message: impl Into<String>,
        host: Option<String>,
        shader: Option<String>,
        kernel: Option<String>,
    ) -> Self {
        Self {
            severity: severity.to_string(),
            check: check.to_string(),
            message: message.into(),
            host,
            shader,
            kernel,
            extra: Map::new(),
        }
    }

    fn extra(mut self, extra: Map<String, Value>) -> Self {
        self.extra = extra;
        self
    }

    fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("severity".into(), json!(self.severity));
        m.insert("check".into(), json!(self.check));
        m.insert("message".into(), json!(self.message));
        m.insert("host".into(), json!(self.host));
        m.insert("shader".into(), json!(self.shader));
        m.insert("kernel".into(), json!(self.kernel));
        if !self.extra.is_empty() {
            m.insert("extra".into(), Value::Object(self.extra.clone()));
        }
        Value::Object(m)
    }
}

pub struct AnalyzeRaw {
    kernels: Vec<MetalKernel>,
    dispatches: Vec<HostDispatch>,
    findings: Vec<Finding>,
    referenced: BTreeMap<String, Vec<String>>,
    metal_names: HashSet<String>,
    generated_kernel_names: BTreeMap<String, BTreeMap<String, String>>,
    family_named: Vec<String>,
    binding_checked: i64,
    geometry_checked: i64,
    structs_paired: i64,
    queue_identity: Value,
    library_membership: BTreeMap<String, String>,
    shaders_not_in_library: Vec<String>,
    counts: BTreeMap<String, i64>,
}

fn floor_char(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn char_index(s: &str, byte_off: usize) -> usize {
    let byte_off = floor_char(s, byte_off.min(s.len()));
    s[..byte_off].chars().count()
}

fn line_of(src: &str, byte_off: usize) -> usize {
    // Python re.Match.start() is a Unicode scalar index; count newlines in
    // the first N scalars of the original source so we match CPython.
    line_of_chars(src, char_index(src, byte_off))
}

fn line_of_chars(src: &str, char_off: usize) -> usize {
    src.chars().take(char_off).filter(|&c| c == '\n').count() + 1
}

fn py_repr_str(s: &str) -> String {
    format!("'{s}'")
}

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

fn py_list_str(xs: &[String]) -> String {
    let inner = xs
        .iter()
        .map(|s| py_repr_str(s))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn loc(path: &str, line: usize) -> String {
    format!("{path}:{line}")
}

pub fn strip_comments_preserve_lines(src: &str) -> String {
    // Character-step to match CPython `src[i]` indexing. `//` consumes the two
    // slashes without emitting them (length shrinks by 2 per line comment).
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let nxt = if i + 1 < n { chars[i + 1] } else { '\0' };
        if c == '/' && nxt == '/' {
            i += 2;
            while i < n && chars[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        if c == '/' && nxt == '*' {
            i += 2;
            out.push(' ');
            out.push(' ');
            while i < n {
                if chars[i] == '*' && i + 1 < n && chars[i + 1] == '/' {
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                    break;
                }
                out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            continue;
        }
        if c == '"' {
            out.push(c);
            i += 1;
            while i < n {
                out.push(chars[i]);
                if chars[i] == '\\' && i + 1 < n {
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                if chars[i] == '"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn match_paren(src: &str, open_pos: usize) -> isize {
    let bytes = src.as_bytes();
    if open_pos >= bytes.len() {
        return -1;
    }
    let opener = bytes[open_pos];
    let closer = match opener {
        b'(' => b')',
        b'{' => b'}',
        b'[' => b']',
        _ => return -1,
    };
    let mut depth = 0i32;
    let mut i = open_pos;
    let n = bytes.len();
    let mut in_str = false;
    let mut str_ch = 0u8;
    while i < n {
        let c = bytes[i];
        if in_str {
            if c == b'\\' && i + 1 < n {
                i += 2;
                continue;
            }
            if c == str_ch {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'\'' {
            in_str = true;
            str_ch = c;
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            i += 2;
            while i < n && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        if c == opener {
            depth += 1;
        } else if c == closer {
            depth -= 1;
            if depth == 0 {
                return i as isize;
            }
        }
        i += 1;
    }
    -1
}

fn split_top_level(src: &str, sep: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut depth_paren = 0i32;
    let mut depth_brack = 0i32;
    let mut depth_brace = 0i32;
    let mut in_str = false;
    let mut str_ch = '\0';
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if in_str {
            buf.push(c);
            if c == '\\' && i + 1 < n {
                buf.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if c == str_ch {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == '"' || c == '\'' {
            in_str = true;
            str_ch = c;
            buf.push(c);
            i += 1;
            continue;
        }
        match c {
            '(' => depth_paren += 1,
            ')' => depth_paren -= 1,
            '[' => depth_brack += 1,
            ']' => depth_brack -= 1,
            '{' => depth_brace += 1,
            '}' => depth_brace -= 1,
            _ => {}
        }
        if c == sep && depth_paren == 0 && depth_brack == 0 && depth_brace == 0 {
            parts.push(buf.clone());
            buf.clear();
            i += 1;
            continue;
        }
        buf.push(c);
        i += 1;
    }
    if !buf.trim().is_empty() {
        parts.push(buf);
    }
    parts
}

fn string_literals(src: &str) -> Vec<String> {
    STRING_LIT_RE
        .captures_iter(src)
        .map(|c| c.get(1).unwrap().as_str().to_string())
        .collect()
}

fn metal_prim(t: &str) -> Option<(i64, i64)> {
    Some(match t {
        "bool" | "uchar" | "char" => (1, 1),
        "ushort" | "short" | "half" => (2, 2),
        "uint" | "int" | "float" => (4, 4),
        "ulong" | "long" | "double" => (8, 8),
        "float2" | "uint2" | "int2" => (8, 8),
        "half2" => (4, 4),
        "uchar2" => (2, 2),
        "float3" | "float4" | "uint3" | "uint4" | "int3" | "int4" => (16, 16),
        "half4" | "ushort4" | "half3" => (8, 8),
        "uchar4" => (4, 4),
        _ => return None,
    })
}

fn rust_prim(t: &str) -> Option<(i64, i64)> {
    Some(match t {
        "bool" | "u8" | "i8" => (1, 1),
        "u16" | "i16" | "f16" => (2, 2),
        "u32" | "i32" | "f32" => (4, 4),
        "u64" | "i64" | "f64" | "usize" | "isize" => (8, 8),
        _ => return None,
    })
}

fn align_up(n: i64, a: i64) -> i64 {
    if a <= 1 {
        return n;
    }
    let r = n % a;
    if r == 0 {
        n
    } else {
        n + (a - r)
    }
}

fn metal_field_size_align(type_s: &str) -> Option<(i64, i64)> {
    let t = WS_RE.replace_all(type_s.trim(), " ").into_owned();
    if t.contains('*')
        || t.starts_with("device ")
        || t.starts_with("constant ")
        || t.starts_with("threadgroup ")
    {
        if t.contains('*') {
            return Some((8, 8));
        }
    }
    if let Some(caps) = METAL_ARRAY_RE.captures(&t) {
        let inner = metal_field_size_align(caps.get(1).unwrap().as_str())?;
        let n: i64 = caps.get(2).unwrap().as_str().parse().ok()?;
        return Some((inner.0 * n, inner.1));
    }
    if let Some(v) = metal_prim(&t) {
        return Some(v);
    }
    let base = t.split(' ').last().unwrap_or("");
    metal_prim(base)
}

fn rust_field_size_align(type_s: &str) -> Option<(i64, i64)> {
    let t = WS_RE.replace_all(type_s.trim(), " ").into_owned();
    if t.starts_with('*') {
        return Some((8, 8));
    }
    if let Some(caps) = RUST_ARRAY_RE.captures(&t) {
        let inner = rust_field_size_align(caps.get(1).unwrap().as_str())?;
        let n: i64 = caps.get(2).unwrap().as_str().parse().ok()?;
        return Some((inner.0 * n, inner.1));
    }
    if let Some(v) = rust_prim(&t) {
        return Some(v);
    }
    let last = t.split("::").last().unwrap_or(&t);
    rust_prim(last).or_else(|| rust_prim(&t))
}

fn layout_fields(
    fields: &[(String, String)],
    rust_lang: bool,
    nested: &HashMap<String, StructDef>,
) -> Option<(Vec<LayoutRow>, i64, i64)> {
    let mut off = 0i64;
    let mut max_a = 1i64;
    let mut rows = Vec::new();
    for (name, ty) in fields {
        let mut sa = if rust_lang {
            rust_field_size_align(ty)
        } else {
            metal_field_size_align(ty)
        };
        if sa.is_none() {
            let key = ty.split("::").last().unwrap_or(ty).trim();
            if let Some(st) = nested.get(key) {
                if let (Some(sz), Some(al)) = (st.size, st.align) {
                    sa = Some((sz, al));
                }
            }
        }
        let (sz, al) = sa?;
        off = align_up(off, al);
        rows.push(LayoutRow {
            name: name.clone(),
            type_s: ty.clone(),
            offset: off,
            size: sz,
        });
        off += sz;
        max_a = max_a.max(al);
    }
    let total = align_up(off, max_a);
    Some((rows, total, max_a))
}

fn classify_shader_param(type_s: &str, attr: &str) -> (String, String, Option<i64>, String) {
    let mut space = "builtin".to_string();
    let mut index = None;
    let attr_s = attr.trim();
    let head = attr_s
        .split('(')
        .next()
        .unwrap_or(attr_s)
        .trim()
        .to_string();
    if attr_s.contains("buffer(") {
        let lead_split = type_s.split('*').next().unwrap_or(type_s);
        space = if lead_split.contains("constant") {
            "constant".into()
        } else {
            "device".into()
        };
        let lead = type_s.trim();
        if lead.starts_with("constant") {
            space = "constant".into();
        } else if lead.starts_with("device") || lead.starts_with("const device") {
            space = "device".into();
        }
        if let Some(c) = BUF_IDX_RE.captures(attr_s) {
            index = c.get(1).and_then(|m| m.as_str().parse().ok());
        }
    } else if attr_s.contains("threadgroup(") {
        space = "threadgroup".into();
        if let Some(c) = TG_IDX_RE.captures(attr_s) {
            index = c.get(1).and_then(|m| m.as_str().parse().ok());
        }
    }
    let mut kind = KIND_UNKNOWN.to_string();
    if space == "threadgroup" {
        kind = KIND_THREADGROUP.into();
    } else if space == "device" {
        kind = KIND_DEVICE.into();
    } else if space == "constant" {
        let t = type_s
            .replace("constant", "")
            .replace('&', "")
            .replace("const", "");
        let t = WS_RE.replace_all(t.trim(), " ").into_owned();
        if t == "uint" || t == "int" {
            kind = KIND_CONSTANT_U32.into();
        } else if t == "float" || t == "half" {
            kind = KIND_CONSTANT_F32.into();
        } else if type_s.contains('*') {
            kind = KIND_DEVICE.into();
        } else {
            kind = KIND_CONSTANT_STRUCT.into();
        }
    }
    (space, kind, index, head)
}

fn parse_metal(src: &str, path: &str) -> (Vec<MetalKernel>, Vec<StructDef>) {
    let clean = strip_comments_preserve_lines(src);
    let mut kernels = Vec::new();
    for m in KERNEL_RE.captures_iter(&clean) {
        let name = m.get(1).unwrap().as_str().to_string();
        let full = m.get(0).unwrap();
        let open_p = full.end() - 1;
        let close_p = match_paren(&clean, open_p);
        if close_p < 0 {
            continue;
        }
        let body = &clean[open_p + 1..close_p as usize];
        let mut params = Vec::new();
        for raw in split_top_level(body, ',') {
            let chunk = raw.trim();
            if chunk.is_empty() {
                continue;
            }
            let Some(am) = ATTR_RE.captures(chunk) else {
                continue;
            };
            let attr = am.get(1).unwrap().as_str();
            let before = chunk[..am.get(0).unwrap().start()]
                .trim()
                .trim_end_matches(',')
                .trim();
            let (pname, type_s) = if let Some(nm) = PARAM_NAME_RE.captures(before) {
                let p = nm.get(1).unwrap().as_str().to_string();
                let t = before[..nm.get(0).unwrap().start()].trim().to_string();
                (p, t)
            } else {
                (String::new(), before.to_string())
            };
            let (space, kind, index, _head) = classify_shader_param(&type_s, attr);
            params.push(ShaderParam {
                name: pname,
                type_s,
                space,
                index,
                kind,
            });
        }
        let mut k = MetalKernel {
            name,
            path: path.to_string(),
            line: line_of_chars(src, char_index(&clean, full.start())),
            buffer_indices: BTreeMap::new(),
            threadgroup_indices: BTreeMap::new(),
        };
        for p in params {
            if (p.space == "device" || p.space == "constant") && p.index.is_some() {
                k.buffer_indices.insert(p.index.unwrap(), p);
            } else if p.space == "threadgroup" && p.index.is_some() {
                k.threadgroup_indices.insert(p.index.unwrap(), p);
            }
        }
        kernels.push(k);
    }
    let mut structs = Vec::new();
    for m in STRUCT_RE.captures_iter(&clean) {
        let name = m.get(1).unwrap().as_str().to_string();
        let full = m.get(0).unwrap();
        let open_b = full.end() - 1;
        let close_b = match_paren(&clean, open_b);
        if close_b < 0 {
            continue;
        }
        let body = &clean[open_b + 1..close_b as usize];
        let mut fields = Vec::new();
        for raw in body.split(';') {
            let chunk = raw.trim();
            if chunk.is_empty() || chunk.starts_with('{') {
                continue;
            }
            let Some(fm) = FIELD_NAME_RE.captures(chunk) else {
                continue;
            };
            let fname = fm.get(1).unwrap().as_str();
            if matches!(fname, "struct" | "enum" | "if" | "for" | "return") {
                continue;
            }
            let mut type_s = chunk[..fm.get(0).unwrap().start()].trim().to_string();
            if type_s.is_empty() {
                continue;
            }
            if let Some(n) = fm.get(2) {
                type_s = format!("{}[{}]", type_s, n.as_str());
            }
            fields.push(StructField {
                name: fname.to_string(),
                type_s,
            });
        }
        if fields.is_empty() {
            continue;
        }
        structs.push(StructDef {
            name,
            path: path.to_string(),
            line: line_of_chars(src, char_index(&clean, full.start())),
            fields,
            lang: "metal".into(),
            size: None,
            align: None,
            layout: None,
        });
    }
    (kernels, structs)
}

fn generated_kernel_names(
    metal_files: &BTreeMap<String, String>,
) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut generated = BTreeMap::new();
    for (path, src) in metal_files {
        if !path.ends_with("qwen_uniform_q4.metal") {
            continue;
        }
        for m in QWEN_K.captures_iter(src) {
            let name = format!(
                "qwen_uniform_q4_group64_matmul_k{}_geo_tpr64_tg128",
                m.get(1).unwrap().as_str()
            );
            let mut rec = BTreeMap::new();
            rec.insert("path".into(), path.clone());
            rec.insert("macro".into(), "QWEN_UNIFORM_Q4_MATMUL_K".into());
            rec.insert(
                "invocation".into(),
                m.get(0).unwrap().as_str().trim().to_string(),
            );
            rec.insert(
                "verification".into(),
                "NAME_RECOVERED_FROM_MACRO; body binding and PSO geometry remain compiler/runtime checks".into(),
            );
            generated.insert(name, rec);
        }
        for m in QWEN_RK.captures_iter(src) {
            let name = format!(
                "qwen_uniform_q4_group64_matmul_r{}k{}_geo_tpr64_tg128",
                m.get(1).unwrap().as_str(),
                m.get(2).unwrap().as_str()
            );
            let mut rec = BTreeMap::new();
            rec.insert("path".into(), path.clone());
            rec.insert("macro".into(), "QWEN_UNIFORM_Q4_MATMUL_RK".into());
            rec.insert(
                "invocation".into(),
                m.get(0).unwrap().as_str().trim().to_string(),
            );
            rec.insert(
                "verification".into(),
                "NAME_RECOVERED_FROM_MACRO; body binding and PSO geometry remain compiler/runtime checks".into(),
            );
            generated.insert(name, rec);
        }
        for m in QWEN_BP.captures_iter(src) {
            let name = format!(
                "qwen_binary_planes_k{}_matvec_geo_tpr64_tg128",
                m.get(1).unwrap().as_str()
            );
            let mut rec = BTreeMap::new();
            rec.insert("path".into(), path.clone());
            rec.insert("macro".into(), "QWEN_BINARY_PLANES".into());
            rec.insert(
                "invocation".into(),
                m.get(0).unwrap().as_str().trim().to_string(),
            );
            rec.insert(
                "verification".into(),
                "NAME_RECOVERED_FROM_MACRO; body binding and PSO geometry remain compiler/runtime checks".into(),
            );
            generated.insert(name, rec);
        }
    }
    generated
}

fn enclosing_fn(src: &str, pos: usize) -> (usize, usize, String, Option<String>) {
    for m in FN_RE.find_iter(src) {
        if m.start() > pos {
            break;
        }
        let brace = match src[m.end()..].find('{') {
            Some(off) => m.end() + off,
            None => continue,
        };
        if brace > pos {
            continue;
        }
        let end = match_paren(src, brace);
        if end >= pos as isize {
            let best = m.start();
            let fn_end = end as usize;
            let fn_src = src[best..=fn_end].to_string();
            let pre_start = floor_char(src, best.saturating_sub(400));
            let pre = &src[pre_start..best];
            let mut cfg = None;
            if let Some(last) = CFG_FEATURE_RE.captures_iter(pre).last() {
                cfg = last
                    .get(1)
                    .or_else(|| last.get(2))
                    .map(|g| g.as_str().to_string());
            }
            return (best, fn_end, fn_src, cfg);
        }
    }
    (0, src.len(), src.to_string(), None)
}

fn parse_triple(
    expr: &str,
    consts: &HashMap<String, i64>,
) -> (Option<i64>, Option<i64>, Option<i64>) {
    let e = expr.trim();
    if !(e.starts_with('(') && e.ends_with(')')) {
        return (None, None, None);
    }
    let parts = split_top_level(&e[1..e.len() - 1], ',');
    if parts.len() != 3 {
        return (None, None, None);
    }
    let one = |p: &str| -> Option<i64> {
        let mut s = p.trim().to_string();
        s = AS_U32_RE.replace(&s, "").into_owned();
        if DIGITS_RE.is_match(&s) {
            return s.parse().ok();
        }
        if let Some(v) = consts.get(&s) {
            return Some(*v);
        }
        INT_LIT_RE
            .captures(&s)
            .and_then(|c| c.get(1).unwrap().as_str().parse().ok())
    };
    (one(&parts[0]), one(&parts[1]), one(&parts[2]))
}

fn resolve_kernel_expr(
    expr: &str,
    fn_src: &str,
    file_const_str: &HashMap<String, String>,
    pick_fns: &HashMap<String, (String, String)>,
    metal_names: &HashSet<String>,
) -> (Vec<String>, String) {
    let e = expr.trim().trim_end_matches(',').trim();
    if let Some(c) = KERNEL_NAME_LIT_RE.captures(e) {
        return (
            vec![c.get(1).unwrap().as_str().to_string()],
            "resolved".into(),
        );
    }
    if let Some(c) = DECODE_FAMILY_CALL_RE.captures(e) {
        let fn_name = c.get(1).unwrap().as_str();
        if let Some((a, b)) = pick_fns.get(fn_name) {
            let mut names = Vec::new();
            for key in [a.as_str(), b.as_str()] {
                if let Some(v) = file_const_str.get(key) {
                    names.push(v.clone());
                } else if metal_names.contains(key) {
                    names.push(key.to_string());
                }
            }
            if names.is_empty() {
                return (vec![fn_name.to_string()], "resolved".into());
            }
            let status = if names.len() > 1 { "dual" } else { "resolved" };
            return (names, status.into());
        }
        if let Some(v) = file_const_str.get(fn_name) {
            return (vec![v.clone()], "resolved".into());
        }
    }
    if matches!(e, "fn_name" | "first_name" | "second_name") {
        return (vec![], "plumbing".into());
    }
    if UPPER_IDENT_RE.is_match(e) {
        for cm in CONST_STR_RE.captures_iter(fn_src) {
            if cm.get(1).unwrap().as_str() == e {
                return (
                    vec![cm.get(2).unwrap().as_str().to_string()],
                    "resolved".into(),
                );
            }
        }
        if let Some(v) = file_const_str.get(e) {
            return (vec![v.clone()], "resolved".into());
        }
    }
    if IDENT_RE.is_match(e) {
        for cm in CONST_STR_RE.captures_iter(fn_src) {
            if cm.get(1).unwrap().as_str() == e {
                return (
                    vec![cm.get(2).unwrap().as_str().to_string()],
                    "resolved".into(),
                );
            }
        }
        let pat = format!(
            r"(?:let\s+(?:mut\s+)?(?:\([^;]*\b{e}\b[^;]*\)|{e})\s*(?::[^=]+)?\s*=|{e}\s*=)"
        );
        let Ok(assign_re) = Regex::new(&pat) else {
            return (vec![], "unverifiable".into());
        };
        let mut assigned = Vec::new();
        for m in assign_re.find_iter(fn_src) {
            let rest = &fn_src[m.end()..];
            let take = match rest.find(';') {
                Some(i) => i + 1,
                None => floor_char(rest, 2500.min(rest.len())),
            };
            for s in string_literals(&rest[..take]) {
                if metal_names.contains(&s) {
                    assigned.push(s);
                }
            }
        }
        assigned.sort();
        assigned.dedup();
        if !assigned.is_empty() {
            let status = if assigned.len() > 1 {
                "dual"
            } else {
                "resolved"
            };
            return (assigned, status.into());
        }
        return (vec![], "unverifiable".into());
    }
    (vec![], "unverifiable".into())
}

fn index_from_raw(raw: &str) -> Option<i64> {
    let mut s = raw.trim().to_string();
    s = AS_U_RE.replace(&s, "").into_owned();
    INT_LIT_RE
        .captures(&s)
        .and_then(|c| c.get(1).unwrap().as_str().parse().ok())
}

fn extract_binds(closure_src: &str, closure_abs_start: usize, src: &str) -> (Vec<HostBind>, bool) {
    let mut binds = Vec::new();
    let mut unverifiable = false;
    for m in METHOD_BIND_RE.captures_iter(closure_src) {
        let recv = m.name("recv").unwrap().as_str();
        let meth = m.name("meth").unwrap().as_str();
        let raw = m.name("idx").unwrap().as_str();
        if ARGBUF_RECV.contains(&recv) && (meth.ends_with("set_u32") || meth.ends_with("set_f32")) {
            continue;
        }
        let mut kind = if meth == "set_buffer" {
            KIND_DEVICE
        } else if meth.ends_with("set_u32") {
            KIND_CONSTANT_U32
        } else if meth.ends_with("set_f32") {
            KIND_CONSTANT_F32
        } else if meth == "set_bytes" {
            KIND_CONSTANT_BYTES
        } else {
            KIND_THREADGROUP
        };
        let mut hint = String::new();
        let open_paren = closure_src[m.get(0).unwrap().start()..]
            .find('(')
            .map(|i| m.get(0).unwrap().start() + i);
        if let Some(op) = open_paren {
            let close_paren = match_paren(closure_src, op);
            if close_paren >= 0 {
                let call = &closure_src[op..=close_paren as usize];
                if meth == "set_buffer" && call.contains("handle()") {
                    hint = "argbuf".into();
                    kind = KIND_CONSTANT_STRUCT;
                }
            }
        }
        let idx = index_from_raw(raw);
        if idx.is_none() {
            unverifiable = true;
        }
        let line = line_of(src, closure_abs_start + m.get(0).unwrap().start());
        binds.push(HostBind {
            kind: kind.into(),
            index: idx,
            line,
            hint,
        });
    }
    // FREE_BIND without lookbehind: set_u32/set_f32/set_params at a non-ident/dot boundary.
    for (meth, kind) in [
        ("set_u32", KIND_CONSTANT_U32),
        ("set_f32", KIND_CONSTANT_F32),
        ("set_params", KIND_CONSTANT_BYTES),
    ] {
        let mut search_from = 0;
        while let Some(rel) = closure_src[search_from..].find(meth) {
            let abs = search_from + rel;
            let prev_ok = abs == 0 || {
                let p = closure_src.as_bytes()[abs - 1];
                !(p == b'.' || p.is_ascii_alphanumeric() || p == b'_')
            };
            search_from = abs + meth.len();
            if !prev_ok {
                continue;
            }
            let after = &closure_src[abs + meth.len()..];
            let trimmed = after.trim_start();
            if !trimmed.starts_with('(') {
                continue;
            }
            let open_off = (after.len() - trimmed.len()) + (abs + meth.len());
            let close = match_paren(closure_src, open_off);
            if close < 0 {
                continue;
            }
            let inside = &closure_src[open_off + 1..close as usize];
            let parts = split_top_level(inside, ',');
            if parts.len() < 2 {
                continue;
            }
            let raw = parts[1].as_str();
            let idx = index_from_raw(raw);
            if idx.is_none() {
                unverifiable = true;
            }
            let line = line_of(src, closure_abs_start + abs);
            binds.push(HostBind {
                kind: kind.into(),
                index: idx,
                line,
                hint: String::new(),
            });
        }
    }
    if binds.is_empty() && ANY_CALL_RE.is_match(closure_src) {
        unverifiable = true;
    }
    (binds, unverifiable)
}

fn arglayouts_in(fn_src: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for m in ARGLAYOUT_RE.captures_iter(fn_src) {
        let names: Vec<String> = ARGLAYOUT_NAME_RE
            .captures_iter(m.get(1).unwrap().as_str())
            .map(|c| c.get(1).unwrap().as_str().to_string())
            .collect();
        if !names.is_empty() {
            out.push(names);
        }
    }
    out
}

fn parse_rust_host(
    src: &str,
    path: &str,
    metal_names: &HashSet<String>,
    file_const_str: &mut HashMap<String, String>,
    pick_fns: &mut HashMap<String, (String, String)>,
) -> (Vec<HostDispatch>, Vec<StructDef>, Vec<String>) {
    for m in CONST_STR_RE.captures_iter(src) {
        file_const_str.insert(
            m.get(1).unwrap().as_str().to_string(),
            m.get(2).unwrap().as_str().to_string(),
        );
    }
    for m in PICK_FN_RE.captures_iter(src) {
        pick_fns.insert(
            m.get(1).unwrap().as_str().to_string(),
            (
                m.get(2).unwrap().as_str().to_string(),
                m.get(3).unwrap().as_str().to_string(),
            ),
        );
    }
    let mut includes: Vec<String> = INCLUDE_STR_SHADER_RE
        .captures_iter(src)
        .map(|m| m.get(1).unwrap().as_str().to_string())
        .collect();
    includes.extend(
        SHADERS_METAL_RE
            .captures_iter(src)
            .map(|m| m.get(1).unwrap().as_str().to_string()),
    );
    includes.sort();
    includes.dedup();

    let mut structs = Vec::new();
    for m in REPR_C_STRUCT_RE.captures_iter(src) {
        let name = m.get(1).unwrap().as_str().to_string();
        let full = m.get(0).unwrap();
        let open_b = match src[full.end().saturating_sub(1)..].find('{') {
            Some(off) => full.end().saturating_sub(1) + off,
            None => continue,
        };
        let close_b = match_paren(src, open_b);
        if close_b < 0 {
            continue;
        }
        let body = strip_comments_preserve_lines(&src[open_b + 1..close_b as usize]);
        let mut fields = Vec::new();
        for raw in split_top_level(&body, ',') {
            let chunk = raw.trim().trim_end_matches(',').trim();
            if chunk.is_empty() || chunk.starts_with('#') || chunk.starts_with("//") {
                continue;
            }
            let Some(fm) = RUST_FIELD_RE.captures(chunk) else {
                continue;
            };
            let fname = fm.get(1).unwrap().as_str();
            if matches!(fname, "fn" | "impl" | "const") {
                continue;
            }
            let ty = WS_RE
                .replace_all(fm.get(2).unwrap().as_str().trim(), " ")
                .into_owned();
            fields.push(StructField {
                name: fname.to_string(),
                type_s: ty,
            });
        }
        if !fields.is_empty() {
            structs.push(StructDef {
                name,
                path: path.to_string(),
                line: line_of(src, full.start()),
                fields,
                lang: "rust".into(),
                size: None,
                align: None,
                layout: None,
            });
        }
    }

    let mut dispatches = Vec::new();
    for m in DISPATCH_RE.find_iter(src) {
        let method_raw = m
            .as_str()
            .trim_start_matches('.')
            .split('(')
            .next()
            .unwrap_or("")
            .trim();
        let method = method_raw.to_string();
        let open_p = m.end() - 1;
        let close_p = match_paren(src, open_p);
        if close_p < 0 {
            continue;
        }
        let args_src = &src[open_p + 1..close_p as usize];
        let args: Vec<String> = split_top_level(args_src, ',')
            .into_iter()
            .map(|a| a.trim().to_string())
            .collect();
        let groups: Vec<Vec<String>> = if method.ends_with("pair_in_one_encoder") {
            let mut g = Vec::new();
            if args.len() >= 4 {
                g.push(args[0..4].to_vec());
            }
            if args.len() >= 9 {
                g.push(args[5..9].to_vec());
            }
            g
        } else if args.len() >= 3 {
            vec![args.iter().take(4).cloned().collect()]
        } else {
            vec![]
        };
        let (fn_start, _fn_end, fn_src, cfg) = enclosing_fn(src, m.start());
        let mut local_consts: HashMap<String, i64> = CONST_U32_RE
            .captures_iter(&fn_src)
            .filter_map(|c| {
                Some((
                    c.get(1).unwrap().as_str().to_string(),
                    c.get(2).unwrap().as_str().parse().ok()?,
                ))
            })
            .collect();
        let prefix_end = floor_char(src, (fn_start + 1).min(src.len()));
        let window_start = floor_char(src, prefix_end.saturating_sub(4000));
        for c in CONST_U32_RE.captures_iter(&src[window_start..prefix_end]) {
            local_consts
                .entry(c.get(1).unwrap().as_str().to_string())
                .or_insert_with(|| c.get(2).unwrap().as_str().parse().unwrap_or(0));
        }
        let arglayouts = arglayouts_in(&fn_src);
        for g in groups {
            if g.len() < 3 {
                continue;
            }
            let kexpr = &g[0];
            let grid_raw = &g[1];
            let tg_raw = &g[2];
            let encode_arg = if g.len() > 3 { g[3].as_str() } else { "" };
            if kexpr.contains("MTLSize") {
                continue;
            }
            if matches!(
                encode_arg.trim(),
                "encode" | "first_encode" | "second_encode"
            ) {
                continue;
            }
            let fn_prefix = &src[fn_start..m.start()];
            let (resolved, status) =
                resolve_kernel_expr(kexpr, fn_prefix, file_const_str, pick_fns, metal_names);
            if status == "plumbing" {
                continue;
            }
            let mut binds = Vec::new();
            let mut binds_uv = false;
            if let Some(cm) = CLOSURE_RE.captures(encode_arg) {
                let body = if cm.get(1).is_some() {
                    let copen = encode_arg.find('{').unwrap_or(0);
                    let cclose = match_paren(encode_arg, copen);
                    if cclose >= 0 {
                        encode_arg[copen..=cclose as usize].to_string()
                    } else {
                        encode_arg.to_string()
                    }
                } else {
                    encode_arg.to_string()
                };
                let abs_start = src[open_p..]
                    .find(encode_arg)
                    .map(|i| open_p + i)
                    .unwrap_or(m.start());
                let (b, uv) = extract_binds(&body, abs_start, src);
                binds = b;
                binds_uv = uv;
            } else if let Some(helper) = HELPER_CALL_RE.captures(encode_arg) {
                let hname = helper.get(1).unwrap().as_str();
                let fn_pat = format!(r"fn\s+{}\s*\(", regex::escape(hname));
                if let Ok(hre) = Regex::new(&fn_pat) {
                    if let Some(hm) = hre.find(src) {
                        let (_, _, hsrc, _) = enclosing_fn(src, hm.start());
                        let (b, uv) = extract_binds(&hsrc, hm.start(), src);
                        binds = b;
                        binds_uv = uv;
                    } else {
                        binds_uv = true;
                    }
                } else {
                    binds_uv = true;
                }
            } else if !encode_arg.trim().is_empty() {
                binds_uv = true;
            }
            let mut argbuf_map = BTreeMap::new();
            let ab_binds: Vec<&HostBind> = binds
                .iter()
                .filter(|b| b.hint == "argbuf" && b.index.is_some())
                .collect();
            for (layout, b) in arglayouts.iter().zip(ab_binds.iter()) {
                if let Some(idx) = b.index {
                    argbuf_map.insert(idx, layout.clone());
                }
            }
            dispatches.push(HostDispatch {
                path: path.to_string(),
                line: line_of(src, m.start()),
                kernel_expr: {
                    let s = kexpr.trim();
                    s.chars().take(160).collect()
                },
                resolved,
                resolve_status: status,
                grid_raw: grid_raw.trim().chars().take(160).collect(),
                tg_raw: tg_raw.trim().chars().take(160).collect(),
                grid: parse_triple(grid_raw, &local_consts),
                tg: parse_triple(tg_raw, &local_consts),
                binds,
                binds_unverifiable: binds_uv,
                feature_cfg: cfg.clone(),
                argbuf_layouts: argbuf_map,
            });
        }
    }
    (dispatches, structs, includes)
}

fn parse_decode_family(
    src: &str,
) -> (
    HashMap<String, String>,
    HashMap<String, (String, String)>,
    Vec<String>,
) {
    let mut consts = HashMap::new();
    for m in CONST_STR_RE.captures_iter(src) {
        consts.insert(
            m.get(1).unwrap().as_str().to_string(),
            m.get(2).unwrap().as_str().to_string(),
        );
    }
    let mut picks = HashMap::new();
    for m in PICK_FN_RE.captures_iter(src) {
        picks.insert(
            m.get(1).unwrap().as_str().to_string(),
            (
                m.get(2).unwrap().as_str().to_string(),
                m.get(3).unwrap().as_str().to_string(),
            ),
        );
    }
    let mut named = Vec::new();
    for m in FAMILY_ARRAY_RE.captures_iter(src) {
        named.extend(
            FAMILY_IDENT_RE
                .captures_iter(m.get(1).unwrap().as_str())
                .map(|c| c.get(1).unwrap().as_str().to_string()),
        );
    }
    let mut kernel_names = Vec::new();
    for n in &named {
        if let Some(v) = consts.get(n) {
            kernel_names.push(v.clone());
        }
    }
    kernel_names.extend(consts.values().cloned());
    kernel_names.sort();
    kernel_names.dedup();
    (consts, picks, kernel_names)
}

pub fn parse_cfg_modules(lib_src: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for m in CFG_MOD_RE.captures_iter(lib_src) {
        let feat = m
            .get(1)
            .or_else(|| m.get(2))
            .map(|g| g.as_str().to_string());
        let name = m.get(3).unwrap().as_str().to_string();
        if let Some(f) = feat {
            out.insert(name, f);
        }
    }
    out
}

pub fn parse_library_membership(metal_mod_src: &str) -> BTreeMap<String, String> {
    let const_to_file: HashMap<String, String> = SHADER_CONST_RE
        .captures_iter(metal_mod_src)
        .map(|m| {
            (
                m.get(1).unwrap().as_str().to_string(),
                m.get(2).unwrap().as_str().to_string(),
            )
        })
        .collect();
    let Some(fn_m) = ALL_SHADER_FN_RE.find(metal_mod_src) else {
        return const_to_file
            .into_iter()
            .map(|(_, f)| (f, "default".into()))
            .collect();
    };
    let brace = match metal_mod_src[fn_m.end().saturating_sub(1)..].find('{') {
        Some(off) => fn_m.end().saturating_sub(1) + off,
        None => {
            return const_to_file
                .into_iter()
                .map(|(_, f)| (f, "default".into()))
                .collect()
        }
    };
    let end = match_paren(metal_mod_src, brace);
    if end < 0 {
        return BTreeMap::new();
    }
    let body = &metal_mod_src[brace..=end as usize];
    let tq_at = body.find("#[cfg(feature = \"tq\")]");
    let default_body = match tq_at {
        Some(i) => &body[..i],
        None => body,
    };
    let tq_body = match tq_at {
        Some(i) => &body[i..],
        None => "",
    };
    let mut membership = BTreeMap::new();
    for (const_name, fn_name) in &const_to_file {
        if tq_body.contains(const_name) && !default_body.contains(const_name) {
            membership.insert(fn_name.clone(), "tq".into());
        } else if default_body.contains(const_name) {
            membership.insert(fn_name.clone(), "default".into());
        }
    }
    membership
}

fn compatible_kinds(host: &str, shader: &str) -> bool {
    if host == shader {
        return true;
    }
    if host == KIND_DEVICE
        && matches!(
            shader,
            KIND_CONSTANT_STRUCT | KIND_CONSTANT_U32 | KIND_CONSTANT_F32 | KIND_CONSTANT_BYTES
        )
    {
        return true;
    }
    if host == KIND_CONSTANT_STRUCT && shader == KIND_CONSTANT_STRUCT {
        return true;
    }
    if host == KIND_CONSTANT_BYTES
        && matches!(
            shader,
            KIND_CONSTANT_U32 | KIND_CONSTANT_F32 | KIND_CONSTANT_STRUCT
        )
    {
        return true;
    }
    if host == KIND_CONSTANT_U32 && shader == KIND_CONSTANT_U32 {
        return true;
    }
    if host == KIND_CONSTANT_F32 && shader == KIND_CONSTANT_F32 {
        return true;
    }
    false
}

fn arglayout_width(x: &str) -> Option<i64> {
    match x {
        "U32" | "F32" => Some(4),
        "U64" => Some(8),
        _ => None,
    }
}

fn abi_field_names_compatible(
    metal_name: &str,
    rust_name: &str,
    metal_ty: &str,
    rust_ty: &str,
) -> bool {
    if metal_name == rust_name {
        return true;
    }
    if metal_name.contains(rust_name) || rust_name.contains(metal_name) {
        return true;
    }
    if metal_ty.contains('*')
        || metal_ty
            .replace("const", " ")
            .split('*')
            .next()
            .unwrap_or("")
            .contains("device")
    {
        let last = rust_ty.split("::").last().unwrap_or(rust_ty);
        if last == "u64" || last == "usize" {
            return true;
        }
    }
    false
}

fn compute_struct_layouts(structs: &mut [StructDef]) {
    for lang in ["metal", "rust"] {
        let rust_lang = lang == "rust";
        loop {
            let nested: HashMap<String, StructDef> = structs
                .iter()
                .filter(|s| s.lang == lang && s.size.is_some())
                .map(|s| (s.name.clone(), s.clone()))
                .collect();
            let mut changed = false;
            for st in structs.iter_mut().filter(|s| s.lang == lang) {
                if st.size.is_some() {
                    continue;
                }
                let pairs: Vec<(String, String)> = st
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), f.type_s.clone()))
                    .collect();
                if let Some((rows, total, al)) = layout_fields(&pairs, rust_lang, &nested) {
                    st.layout = Some(rows);
                    st.size = Some(total);
                    st.align = Some(al);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
}

fn extra_map(pairs: &[(&str, Value)]) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert((*k).into(), v.clone());
    }
    m
}

pub fn analyze(
    metal_files: &BTreeMap<String, String>,
    rust_files: &BTreeMap<String, String>,
    library_membership: Option<&BTreeMap<String, String>>,
    production_host_prefix: &str,
) -> AnalyzeRaw {
    let mut findings: Vec<Finding> = Vec::new();
    let mut kernels = Vec::new();
    let mut metal_structs = Vec::new();
    for (path, src) in metal_files {
        let (ks, sts) = parse_metal(src, path);
        kernels.extend(ks);
        metal_structs.extend(sts);
    }
    let mut by_name: BTreeMap<String, Vec<MetalKernel>> = BTreeMap::new();
    for k in &kernels {
        by_name.entry(k.name.clone()).or_default().push(k.clone());
    }
    let generated_names = generated_kernel_names(metal_files);
    let mut metal_names: HashSet<String> = by_name.keys().cloned().collect();
    metal_names.extend(generated_names.keys().cloned());

    for (name, ks) in &by_name {
        if ks.len() > 1 {
            let shader = ks
                .iter()
                .map(|k| loc(&k.path, k.line))
                .collect::<Vec<_>>()
                .join(", ");
            findings.push(Finding::new(
                "WARNING",
                "duplicate_kernel_name",
                format!(
                    "kernel {} is defined in {} shader files",
                    py_repr_str(name),
                    ks.len()
                ),
                None,
                Some(shader),
                Some(name.clone()),
            ));
        }
    }

    let mut decode_consts = HashMap::new();
    let mut pick_fns: HashMap<String, (String, String)> = HashMap::new();
    let mut family_named = Vec::new();
    for (path, src) in rust_files {
        if path.ends_with("decode_family.rs") {
            let (c, p, n) = parse_decode_family(src);
            decode_consts = c;
            pick_fns = p;
            family_named = n;
            break;
        }
    }
    let mut module_feat = HashMap::new();
    for (path, src) in rust_files {
        if path.ends_with("src/lib.rs") || path.ends_with("/lib.rs") {
            module_feat = parse_cfg_modules(src);
            break;
        }
    }

    let mut all_dispatches = Vec::new();
    let mut rust_structs = Vec::new();
    let mut merged_consts = decode_consts.clone();
    for (path, src) in rust_files {
        let (ds, sts, _includes) =
            parse_rust_host(src, path, &metal_names, &mut merged_consts, &mut pick_fns);
        let stem = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let file_feat = module_feat.get(stem).cloned();
        let mut ds = ds;
        if let Some(ff) = file_feat {
            for d in &mut ds {
                if d.feature_cfg.is_none() {
                    d.feature_cfg = Some(ff.clone());
                }
            }
        }
        all_dispatches.extend(ds);
        rust_structs.extend(sts);
    }

    compute_struct_layouts(&mut metal_structs);
    compute_struct_layouts(&mut rust_structs);

    let mut referenced: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for d in &all_dispatches {
        for n in &d.resolved {
            referenced
                .entry(n.clone())
                .or_default()
                .push(loc(&d.path, d.line));
        }
    }
    for n in &family_named {
        referenced
            .entry(n.clone())
            .or_default()
            .push("crates/hawking-core/src/decode_family.rs:FAMILY".into());
    }
    for (path, src) in rust_files {
        for lit in string_literals(src) {
            if metal_names.contains(&lit) {
                referenced.entry(lit).or_default().push(path.clone());
            }
        }
    }

    let missing_host_refs: Vec<String> = referenced
        .keys()
        .filter(|n| !metal_names.contains(*n))
        .cloned()
        .collect();
    for n in missing_host_refs {
        let sites = referenced.get(&n).cloned().unwrap_or_default();
        let extra = extra_map(&[(
            "host_sites",
            json!(sites.iter().take(8).cloned().collect::<Vec<_>>()),
        )]);
        findings.push(
            Finding::new(
                "ERROR",
                "kernel_existence",
                format!(
                    "host references kernel {} but no shader defines `kernel void {n}(`",
                    py_repr_str(&n)
                ),
                Some(sites.first().cloned().unwrap_or_default()),
                None,
                Some(n),
            )
            .extra(extra),
        );
    }

    let mut binding_checked = 0i64;
    let mut geometry_checked = 0i64;
    for d in &all_dispatches {
        let host_loc = loc(&d.path, d.line);
        if d.resolve_status == "unverifiable" || d.resolved.is_empty() {
            let extra = extra_map(&[
                ("expr", json!(d.kernel_expr)),
                ("status", json!(d.resolve_status)),
            ]);
            findings.push(
                Finding::new(
                    "UNVERIFIABLE",
                    "kernel_name",
                    "dispatch kernel name is not a string literal or resolvable const; not scored as PASS",
                    Some(host_loc),
                    None,
                    Some(d.kernel_expr.clone()),
                )
                .extra(extra),
            );
            continue;
        }
        for kname in &d.resolved {
            let Some(ks) = by_name.get(kname) else {
                continue;
            };
            let shader = &ks[0];
            let shader_loc = loc(&shader.path, shader.line);
            if d.binds_unverifiable || d.binds.iter().any(|b| b.index.is_none()) {
                findings.push(Finding::new(
                    "UNVERIFIABLE",
                    "binding_index",
                    "one or more host binds use a non-literal index or a helper this checker cannot follow; not scored as PASS",
                    Some(host_loc.clone()),
                    Some(shader_loc.clone()),
                    Some(kname.clone()),
                ));
            } else {
                let host_bufs: HashSet<i64> = d
                    .binds
                    .iter()
                    .filter(|b| b.index.is_some() && b.kind != KIND_THREADGROUP)
                    .filter_map(|b| b.index)
                    .collect();
                let shader_bufs = shader.buffer_set();
                let mut missing: Vec<i64> = shader_bufs.difference(&host_bufs).copied().collect();
                missing.sort();
                let mut extra_h: Vec<i64> = host_bufs.difference(&shader_bufs).copied().collect();
                extra_h.sort();
                let off_by_one = host_bufs.len() == shader_bufs.len()
                    && !host_bufs.is_empty()
                    && !shader_bufs.is_empty()
                    && ({
                        let minus: HashSet<i64> = host_bufs.iter().map(|i| i - 1).collect();
                        let plus: HashSet<i64> = host_bufs.iter().map(|i| i + 1).collect();
                        minus == shader_bufs || plus == shader_bufs
                    });
                if !missing.is_empty() || !extra_h.is_empty() {
                    let mut host_sorted: Vec<i64> = host_bufs.iter().copied().collect();
                    host_sorted.sort();
                    let mut shader_sorted: Vec<i64> = shader_bufs.iter().copied().collect();
                    shader_sorted.sort();
                    let mut msg = format!(
                        "host binds buffers {host_sorted:?} but shader declares {shader_sorted:?}"
                    );
                    if off_by_one {
                        msg = format!("off-by-one buffer index: {msg}");
                    }
                    let extra_only = !extra_h.is_empty() && missing.is_empty() && !off_by_one;
                    let sev = if extra_only { "WARNING" } else { "ERROR" };
                    let extra = extra_map(&[
                        ("host_indices", json!(host_sorted)),
                        ("shader_indices", json!(shader_sorted)),
                        ("missing_on_host", json!(missing)),
                        ("extra_on_host", json!(extra_h)),
                        ("off_by_one", json!(off_by_one)),
                    ]);
                    findings.push(
                        Finding::new(
                            sev,
                            "binding_index",
                            msg,
                            Some(host_loc.clone()),
                            Some(shader_loc.clone()),
                            Some(kname.clone()),
                        )
                        .extra(extra),
                    );
                    if extra_only {
                        binding_checked += 1;
                    }
                } else {
                    binding_checked += 1;
                }
                let mut by_host: HashMap<i64, &HostBind> = HashMap::new();
                for b in &d.binds {
                    if let Some(idx) = b.index {
                        if b.kind != KIND_THREADGROUP {
                            by_host.insert(idx, b);
                        }
                    }
                }
                let mut type_ok = true;
                for (idx, sp) in &shader.buffer_indices {
                    let Some(hb) = by_host.get(idx) else {
                        continue;
                    };
                    if !compatible_kinds(&hb.kind, &sp.kind) {
                        type_ok = false;
                        let extra = extra_map(&[
                            ("index", json!(idx)),
                            ("host_kind", json!(hb.kind)),
                            ("shader_kind", json!(sp.kind)),
                        ]);
                        findings.push(
                            Finding::new(
                                "ERROR",
                                "type_width",
                                format!(
                                    "buffer({idx}): host bind kind {} is not compatible with shader {} ({} {})",
                                    hb.kind, sp.kind, py_repr_str(&sp.type_s), sp.name
                                ),
                                Some(loc(&d.path, hb.line)),
                                Some(shader_loc.clone()),
                                Some(kname.clone()),
                            )
                            .extra(extra),
                        );
                    }
                }
                if type_ok && missing.is_empty() {
                    findings.push(Finding::new(
                        "INFO",
                        "type_width",
                        format!("host bind kinds are compatible with shader {kname}"),
                        Some(host_loc.clone()),
                        Some(shader_loc.clone()),
                        Some(kname.clone()),
                    ));
                }
                for (idx, layout) in &d.argbuf_layouts {
                    let Some(sp) = shader.buffer_indices.get(idx) else {
                        continue;
                    };
                    if sp.kind != KIND_CONSTANT_STRUCT {
                        continue;
                    }
                    let st_name = {
                        let s = sp.type_s.replace(['&', '*'], "");
                        let s = s.replace("constant", "").replace("const", "");
                        s.split_whitespace().last().unwrap_or("").to_string()
                    };
                    let ms = metal_structs.iter().find(|s| s.name == st_name);
                    match ms {
                        None | Some(StructDef { layout: None, .. }) => {
                            findings.push(Finding::new(
                                "UNVERIFIABLE",
                                "host_shader_abi",
                                format!(
                                    "argbuf at buffer({idx}) names shader struct {st_name} whose layout could not be fully resolved"
                                ),
                                Some(host_loc.clone()),
                                Some(shader_loc.clone()),
                                Some(kname.clone()),
                            ));
                        }
                        Some(ms) => {
                            let host_w: Vec<i64> =
                                layout.iter().filter_map(|x| arglayout_width(x)).collect();
                            let sh_w: Vec<i64> =
                                ms.layout.as_ref().unwrap().iter().map(|r| r.size).collect();
                            if host_w != sh_w {
                                findings.push(Finding::new(
                                    "ERROR",
                                    "host_shader_abi",
                                    format!(
                                        "KernelArgBuffer layout {} widths {host_w:?} != shader struct {st_name} field widths {sh_w:?}",
                                        py_list_str(layout)
                                    ),
                                    Some(host_loc.clone()),
                                    Some(loc(&ms.path, ms.line)),
                                    Some(kname.clone()),
                                ));
                            } else {
                                findings.push(Finding::new(
                                    "INFO",
                                    "host_shader_abi",
                                    format!(
                                        "KernelArgBuffer {} matches {st_name} field widths",
                                        py_list_str(layout)
                                    ),
                                    Some(host_loc.clone()),
                                    Some(loc(&ms.path, ms.line)),
                                    Some(kname.clone()),
                                ));
                            }
                        }
                    }
                }
            }

            let host_tg_slots: HashSet<i64> = d
                .binds
                .iter()
                .filter(|b| b.kind == KIND_THREADGROUP && b.index.is_some())
                .filter_map(|b| b.index)
                .collect();
            let shader_tg_slots: HashSet<i64> =
                shader.threadgroup_indices.keys().copied().collect();
            if !shader_tg_slots.is_empty() && !d.binds_unverifiable {
                if host_tg_slots != shader_tg_slots && !host_tg_slots.is_superset(&shader_tg_slots)
                {
                    let mut s_slots: Vec<i64> = shader_tg_slots.iter().copied().collect();
                    s_slots.sort();
                    let mut h_slots: Vec<i64> = host_tg_slots.iter().copied().collect();
                    h_slots.sort();
                    findings.push(Finding::new(
                        "WARNING",
                        "threadgroup_memory",
                        format!(
                            "shader threadgroup slots {s_slots:?} vs host set_threadgroup_memory_length {h_slots:?}"
                        ),
                        Some(host_loc.clone()),
                        Some(shader_loc.clone()),
                        Some(kname.clone()),
                    ));
                }
            }

            let (tx, ty, tz) = d.tg;
            let (gx, gy, gz) = d.grid;
            if tx.is_some() && ty.is_some() && tz.is_some() {
                geometry_checked += 1;
                let prod = tx.unwrap() * ty.unwrap() * tz.unwrap();
                if prod == 0 {
                    findings.push(Finding::new(
                        "ERROR",
                        "dispatch_geometry",
                        format!("threadgroup {:?} has a zero axis", d.tg),
                        Some(host_loc.clone()),
                        Some(shader_loc.clone()),
                        Some(kname.clone()),
                    ));
                } else if prod > APPLE_MAX_THREADS_PER_THREADGROUP {
                    let extra = extra_map(&[("threads_per_threadgroup", json!(prod))]);
                    findings.push(
                        Finding::new(
                            "ERROR",
                            "dispatch_geometry",
                            format!(
                                "threadgroup {:?} product {prod} exceeds Apple Silicon device limit {APPLE_MAX_THREADS_PER_THREADGROUP}",
                                d.tg
                            ),
                            Some(host_loc.clone()),
                            Some(shader_loc.clone()),
                            Some(kname.clone()),
                        )
                        .extra(extra),
                    );
                } else if prod % APPLE_SIMDGROUP_WIDTH != 0 {
                    findings.push(Finding::new(
                        "WARNING",
                        "dispatch_geometry",
                        format!(
                            "threadgroup product {prod} is not a multiple of simdgroup width {APPLE_SIMDGROUP_WIDTH}"
                        ),
                        Some(host_loc.clone()),
                        Some(shader_loc.clone()),
                        Some(kname.clone()),
                    ));
                }
            } else {
                let extra =
                    extra_map(&[("tg_raw", json!(d.tg_raw)), ("grid_raw", json!(d.grid_raw))]);
                findings.push(
                    Finding::new(
                        "UNVERIFIABLE",
                        "dispatch_geometry",
                        "threadgroup size is not a literal/const triple; coverage of the problem size is not scored as PASS",
                        Some(host_loc.clone()),
                        Some(shader_loc.clone()),
                        Some(kname.clone()),
                    )
                    .extra(extra),
                );
            }
            if gx.is_some()
                && gy.is_some()
                && gz.is_some()
                && tx.is_some()
                && ty.is_some()
                && tz.is_some()
            {
                let gprod = gx.unwrap() * gy.unwrap() * gz.unwrap();
                if gprod == 0 {
                    findings.push(Finding::new(
                        "ERROR",
                        "dispatch_geometry",
                        format!(
                            "grid {:?} has a zero axis (would launch no threads)",
                            d.grid
                        ),
                        Some(host_loc.clone()),
                        Some(shader_loc.clone()),
                        Some(kname.clone()),
                    ));
                }
            }
        }
    }

    let rust_by: HashMap<String, StructDef> = rust_structs
        .iter()
        .map(|s| (s.name.clone(), s.clone()))
        .collect();
    let metal_by: BTreeMap<String, StructDef> = metal_structs
        .iter()
        .map(|s| (s.name.clone(), s.clone()))
        .collect();
    let mut paired = 0i64;
    for (mname, ms) in &metal_by {
        let mut rs = rust_by.get(mname).cloned();
        if rs.is_none() {
            let cands: Vec<_> = rust_by
                .iter()
                .filter(|(n, _)| mname.ends_with(n.as_str()) && !n.is_empty())
                .map(|(_, s)| s.clone())
                .collect();
            if cands.len() == 1 {
                rs = Some(cands.into_iter().next().unwrap());
            }
        }
        let Some(rs) = rs else {
            continue;
        };
        if ms.layout.is_none() || rs.layout.is_none() {
            let extra = extra_map(&[("rust", json!(rs.name)), ("metal", json!(ms.name))]);
            findings.push(
                Finding::new(
                    "UNVERIFIABLE",
                    "host_shader_abi",
                    format!(
                        "struct {mname} exists on both sides but a nested field could not be sized"
                    ),
                    Some(loc(&rs.path, rs.line)),
                    Some(loc(&ms.path, ms.line)),
                    None,
                )
                .extra(extra),
            );
            continue;
        }
        paired += 1;
        let mut mismatches = Vec::new();
        let ml = ms.layout.as_ref().unwrap();
        let rl = rs.layout.as_ref().unwrap();
        if ml.len() != rl.len() {
            mismatches.push(format!("field count metal={} rust={}", ml.len(), rl.len()));
        }
        let n = ml.len().min(rl.len());
        for i in 0..n {
            let mf = &ml[i];
            let rf = &rl[i];
            if mf.size != rf.size || mf.offset != rf.offset {
                mismatches.push(format!(
                    "field[{i}] metal {}:{} @{}+{} vs rust {}:{} @{}+{}",
                    mf.name, mf.type_s, mf.offset, mf.size, rf.name, rf.type_s, rf.offset, rf.size
                ));
            } else if !abi_field_names_compatible(&mf.name, &rf.name, &mf.type_s, &rf.type_s) {
                mismatches.push(format!(
                    "field[{i}] order/name metal {} vs rust {} (same width {}; field order is ABI)",
                    py_repr_str(&mf.name),
                    py_repr_str(&rf.name),
                    mf.size
                ));
            }
        }
        if ms.size != rs.size {
            mismatches.push(format!("sizeof metal={:?} rust={:?}", ms.size, rs.size));
        }
        if !mismatches.is_empty() {
            let extra = extra_map(&[(
                "mismatches",
                json!(mismatches.iter().take(12).cloned().collect::<Vec<_>>()),
            )]);
            findings.push(
                Finding::new(
                    "ERROR",
                    "host_shader_abi",
                    format!(
                        "ABI drift in {mname}: {}",
                        mismatches
                            .iter()
                            .take(6)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("; ")
                    ),
                    Some(loc(&rs.path, rs.line)),
                    Some(loc(&ms.path, ms.line)),
                    None,
                )
                .extra(extra),
            );
        } else {
            findings.push(Finding::new(
                "INFO",
                "host_shader_abi",
                format!(
                    "repr(C) {} matches metal {} ({} bytes, {} fields)",
                    rs.name,
                    ms.name,
                    ms.size.unwrap_or(0),
                    ml.len()
                ),
                Some(loc(&rs.path, rs.line)),
                Some(loc(&ms.path, ms.line)),
                None,
            ));
        }
    }

    let tq_kernels: Vec<MetalKernel> = kernels
        .iter()
        .filter(|k| {
            Path::new(&k.path).file_name().and_then(|s| s.to_str()) == Some("strand_bitslice.metal")
        })
        .cloned()
        .collect();
    for k in &tq_kernels {
        let sites: Vec<&HostDispatch> = all_dispatches
            .iter()
            .filter(|d| d.resolved.iter().any(|n| n == &k.name))
            .collect();
        if sites.is_empty() {
            if referenced.contains_key(&k.name) {
                continue;
            }
            findings.push(Finding::new(
                "WARNING",
                "feature_gate",
                format!(
                    "tq kernel {} has no statically resolved host dispatch",
                    k.name
                ),
                None,
                Some(loc(&k.path, k.line)),
                Some(k.name.clone()),
            ));
            continue;
        }
        let cfgs: HashSet<Option<String>> = sites.iter().map(|d| d.feature_cfg.clone()).collect();
        let off_reachable = sites.iter().any(|d| d.feature_cfg.as_deref() != Some("tq"));
        let on_reachable = sites
            .iter()
            .any(|d| d.feature_cfg.as_deref() == Some("tq") || d.feature_cfg.is_none());
        if off_reachable
            && sites.iter().any(|d| {
                d.path.starts_with(production_host_prefix) && d.feature_cfg.as_deref() != Some("tq")
            })
        {
            let mut cfg_list: Vec<String> = cfgs
                .iter()
                .map(|c| c.clone().unwrap_or_else(|| "<none>".into()))
                .collect();
            cfg_list.sort();
            let extra = extra_map(&[("cfgs", json!(cfg_list))]);
            findings.push(
                Finding::new(
                    "ERROR",
                    "feature_gate",
                    format!(
                        "kernel {} is dispatched from a production host without #[cfg(feature = \"tq\")], so it is still reachable when the flag is off (or would fail pipeline lookup on a default library)",
                        k.name
                    ),
                    Some(loc(&sites[0].path, sites[0].line)),
                    Some(loc(&k.path, k.line)),
                    Some(k.name.clone()),
                )
                .extra(extra),
            );
        } else {
            let mut cfg_list: Vec<String> = cfgs
                .iter()
                .map(|c| c.clone().unwrap_or_else(|| "<none>".into()))
                .collect();
            cfg_list.sort();
            findings.push(Finding::new(
                "INFO",
                "feature_gate",
                format!(
                    "kernel {} host sites are cfg-gated {}; reachable_when_on={} genuinely_unreachable_when_off={}",
                    k.name,
                    py_list_str(&cfg_list),
                    py_bool(on_reachable),
                    py_bool(!off_reachable)
                ),
                Some(loc(&sites[0].path, sites[0].line)),
                Some(loc(&k.path, k.line)),
                Some(k.name.clone()),
            ));
        }
    }

    if !pick_fns.is_empty() {
        let mut pick_keys: Vec<String> = pick_fns.keys().cloned().collect();
        pick_keys.sort();
        let extra = extra_map(&[("pick_fns", json!(pick_keys))]);
        findings.push(
            Finding::new(
                "INFO",
                "feature_gate",
                "HAWKING_DECODE_FAMILY is an env opt-out (default on) that picks family vs legacy kernel symbols; both names must exist. This is not a Cargo feature.",
                Some("crates/hawking-core/src/decode_family.rs".into()),
                None,
                None,
            )
            .extra(extra),
        );
        let mut pick_items: Vec<_> = pick_fns.iter().collect();
        pick_items.sort_by(|a, b| a.0.cmp(b.0));
        for (fn_name, (a, b)) in pick_items {
            for key in [a.as_str(), b.as_str()] {
                let name = decode_consts
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| key.to_string());
                if !metal_names.contains(&name) {
                    findings.push(Finding::new(
                        "ERROR",
                        "feature_gate",
                        format!(
                            "decode_family::{fn_name}() can pick {} which has no kernel void",
                            py_repr_str(&name)
                        ),
                        Some("crates/hawking-core/src/decode_family.rs".into()),
                        None,
                        Some(name),
                    ));
                }
            }
        }
    }

    let check_library = library_membership.is_some();
    let membership: BTreeMap<String, String> = library_membership.cloned().unwrap_or_default();
    let file_of_kernel: HashMap<String, String> = kernels
        .iter()
        .map(|k| {
            (
                k.name.clone(),
                Path::new(&k.path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string(),
            )
        })
        .collect();
    for d in &all_dispatches {
        if d.resolved.is_empty() {
            continue;
        }
        if !d.path.starts_with(production_host_prefix) {
            continue;
        }
        for kname in &d.resolved {
            let Some(fn_name) = file_of_kernel.get(kname) else {
                continue;
            };
            let mem = membership.get(fn_name);
            if mem.map(|s| s.as_str()) == Some("tq") && d.feature_cfg.as_deref() != Some("tq") {
                findings.push(Finding::new(
                    "ERROR",
                    "control_path",
                    format!(
                        "production host dispatches {kname} from {fn_name} which all_shader_sources only compiles under feature tq, but this site is not cfg-gated tq"
                    ),
                    Some(loc(&d.path, d.line)),
                    Some(fn_name.clone()),
                    Some(kname.clone()),
                ));
            } else if mem.is_none() && check_library {
                findings.push(Finding::new(
                    "ERROR",
                    "control_path",
                    format!(
                        "production host dispatches {kname} from {fn_name} which is not in MetalContext::all_shader_sources; pipeline() would fail"
                    ),
                    Some(loc(&d.path, d.line)),
                    Some(fn_name.clone()),
                    Some(kname.clone()),
                ));
            }
        }
    }

    let queue_identity = json!({
        "construction": "MetalContext::new builds Device::system_default then device.new_command_queue(); one queue per context",
        "per_dispatch_command_buffer": "MetalContext.dispatch_threads creates a new command buffer, encodes one kernel, commits, and waits",
        "fused_command_buffer": "TokenCommandBuffer and dispatch_batch encode many kernels onto one command buffer before commit",
        "concurrent_group": "dispatch_threads_in_concurrent_group shares one compute encoder",
        "ordered_encoder": "enable_ordered_encoder fuses serial dispatches onto one encoder",
        "statically_determinable": "queue construction and the four encode modes are statically visible. Which MetalContext instance a given call uses at runtime, and whether an ordered/concurrent encoder is currently open, is UNVERIFIABLE without execution.",
        "gpu_authority": false,
    });

    let n_pass_like = findings
        .iter()
        .filter(|f| {
            f.severity == "INFO" && (f.check == "type_width" || f.check == "host_shader_abi")
        })
        .count() as i64;
    let n_error = findings.iter().filter(|f| f.severity == "ERROR").count() as i64;
    let n_warn = findings.iter().filter(|f| f.severity == "WARNING").count() as i64;
    let n_uv = findings
        .iter()
        .filter(|f| f.severity == "UNVERIFIABLE")
        .count() as i64;

    let mut shaders_not_in_library = Vec::new();
    if !membership.is_empty() {
        let on_disk: HashSet<String> = metal_files
            .keys()
            .map(|p| {
                Path::new(p)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        let mem_set: HashSet<String> = membership.keys().cloned().collect();
        shaders_not_in_library = on_disk.difference(&mem_set).cloned().collect();
        shaders_not_in_library.sort();
    }

    let mut counts = BTreeMap::new();
    counts.insert("ERROR".into(), n_error);
    counts.insert("WARNING".into(), n_warn);
    counts.insert("UNVERIFIABLE".into(), n_uv);
    counts.insert("INFO".into(), n_pass_like);

    AnalyzeRaw {
        kernels,
        dispatches: all_dispatches,
        findings,
        referenced,
        metal_names,
        generated_kernel_names: generated_names,
        family_named,
        binding_checked,
        geometry_checked,
        structs_paired: paired,
        queue_identity,
        library_membership: membership,
        shaders_not_in_library,
        counts,
    }
}

fn trim_findings(findings: &[Finding]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut uv_by_check: BTreeMap<String, Vec<&Finding>> = BTreeMap::new();
    let mut info: Vec<&Finding> = Vec::new();
    for f in findings {
        if f.severity == "ERROR" || f.severity == "WARNING" {
            out.push(f.to_json());
        } else if f.severity == "UNVERIFIABLE" {
            uv_by_check.entry(f.check.clone()).or_default().push(f);
        } else {
            info.push(f);
        }
    }
    for (check, rows) in &uv_by_check {
        let sample: Vec<Value> = rows.iter().take(12).map(|r| r.to_json()).collect();
        out.push(json!({
            "severity": "UNVERIFIABLE",
            "check": check,
            "message": format!(
                "{} UNVERIFIABLE site(s) for {check} (sample of {})",
                rows.len(),
                12.min(rows.len())
            ),
            "host": Value::Null,
            "shader": Value::Null,
            "kernel": Value::Null,
            "extra": {"count": rows.len(), "sample": sample},
        }));
    }
    let struct_info: Vec<&&Finding> = info
        .iter()
        .filter(|f| f.check == "host_shader_abi" && f.kernel.is_none())
        .collect();
    let other_info: Vec<&&Finding> = info.iter().filter(|f| f.check == "feature_gate").collect();
    for f in struct_info.into_iter().chain(other_info) {
        out.push(f.to_json());
    }
    let mut info_counts: BTreeMap<String, i64> = BTreeMap::new();
    for f in &info {
        *info_counts.entry(f.check.clone()).or_insert(0) += 1;
    }
    if !info_counts.is_empty() {
        let mut extra = Map::new();
        for (k, v) in info_counts {
            extra.insert(k, json!(v));
        }
        out.push(json!({
            "severity": "INFO",
            "check": "info_tally",
            "message": "INFO rows tallied rather than dumped (PASS-like rows are not promotions)",
            "host": Value::Null,
            "shader": Value::Null,
            "kernel": Value::Null,
            "extra": extra,
        }));
    }
    out
}

fn git_query(repo: &Path, args: &[&str]) -> String {
    let mut cmd = Command::new("git");
    cmd.arg("--no-optional-locks").args(args).current_dir(repo);
    match cmd.output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => String::new(),
    }
}

pub fn report_from_analyze(raw: &AnalyzeRaw, repo: &Path) -> Value {
    let errors: Vec<&Finding> = raw
        .findings
        .iter()
        .filter(|f| f.severity == "ERROR")
        .collect();
    let generated_json: BTreeMap<String, BTreeMap<String, String>> =
        raw.generated_kernel_names.clone();
    let membership_files: Vec<Value> = raw
        .library_membership
        .iter()
        .map(|(k, v)| json!([k, v]))
        .collect();
    json!({
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": "Deterministic host/shader ABI preflight. Reads .metal sources and Rust hosts. Emits STATIC_ONLY. Never a hardware measurement.",
        "static_correctness_does_not_prove_speed": STATIC_CORRECTNESS_IS_NOT_SPEED,
        "does_not_substitute_for_protected_measurement": true,
        "evidence_class": "STATIC_ONLY",
        "measurement_states_we_are_not": ["DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
        "head": git_query(repo, &["rev-parse", "HEAD"]),
        "branch": git_query(repo, &["rev-parse", "--abbrev-ref", "HEAD"]),
        "recovered_implementation": {
            "decode_family_kernel_name_tests": "crates/hawking-core/src/decode_family.rs — unit tests that FAMILY_KERNELS and Q80_TILE_KERNELS appear as `kernel void {name}(` in gk_family.metal / q80_mixed_decode.metal. Name presence only; no buffer-index or ABI check.",
            "metal_mod_trace_name_tests": "crates/hawking-core/src/metal/mod.rs — static_kernel_name plus per-family `SHADER_X.contains(kernel void {name}()` tests. Compile/trace-name contract, not host bind vs [[buffer(N)]].",
            "argbuf_layout_comments": "crates/hawking-core/src/metal/argbuf.rs — KernelArgBuffer packs U32/F32/U64 at natural alignment; the shader must declare a packed constant struct at the bound index. No automated checker compared the two.",
            "megakernel_sizeof_guards": "crates/hawking-core/src/kernels/megakernel.rs — const size_of::<MkLayerArgs>() == 120 and MkArgs == 20. Compile-time size, not field-order vs the .metal.",
            "dispatch_ledger": "tools/headless/dispatch_ledger.py (git, not this sparse checkout) — GPU dispatch-count ledger for a sealed 756-dispatch parent. Measurement, not ABI.",
            "accelerator_geometry_tests": "tools/accelerator/test_threadgroup_width.py, test_native_geometry.py, kernel_forge.py — arithmetic and GPU-backed geometry. Not a host/shader buffer-index preflight.",
            "frontier_entry": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json F014 — 'No static kernel/ABI preflight independent of the GPU'. Probe *static_kernel_verify* was absent when the frontier was sealed.",
            "flash_layer46_dispatch_ledger": "receipts/headless/FLASH_LAYER46_DISPATCH_LEDGER.json was named in the lane contract. It is not on disk in this sparse checkout and is not in git HEAD (git cat-file / ls-tree miss). Not treated as evidence it never existed elsewhere; treated as ABSENT here.",
            "adequate_duplicate": "No existing module performed host set_buffer(N) vs shader [[buffer(N)]] comparison, struct field-order ABI, or feature-gate reachability as a STATIC_ONLY receipt. The name-presence tests above are consumed, not forked."
        },
        "gaps_closed": [
            "kernel existence: every resolved host kernel name must have kernel void in a .metal file",
            "binding count and buffer index, including the classic off-by-one",
            "type/width compatibility between host set_u32/set_f32/set_buffer and shader space",
            "threadgroup product against the Apple Silicon 1024 device ceiling",
            "repr(C) vs metal struct field order, size, and alignment where both exist",
            "KernelArgBuffer ArgLayout vs constant shader struct field widths",
            "feature-gate wiring for strand_bitslice (tq) and HAWKING_DECODE_FAMILY pick()",
            "control-path: production host vs MetalContext::all_shader_sources membership",
            "queue identity described statically (one queue per MetalContext; four encode modes)",
            "UNVERIFIABLE is a first-class result: a bind this checker cannot follow is never PASS"
        ],
        "negative_findings": [
            "No Metal runtime, no PSO, no maxTotalThreadsPerThreadgroup from the driver.",
            "FLASH_LAYER46_DISPATCH_LEDGER.json is absent from git HEAD and from this worktree.",
            "tools/accelerator/* and tools/headless/* are not materialized in this sparse checkout; recovered via git show / git ls-tree only.",
            "Bindings set through helpers in another crate, macros, or function pointers are UNVERIFIABLE.",
            "Grid covering a runtime problem size (rows, seq_len, hidden) is UNVERIFIABLE without those values.",
            "This sidecar produces no DIAGNOSTIC_RELATIVE and no PROTECTED_ABSOLUTE."
        ],
        "coverage": {
            "metal_files": raw.kernels.iter().map(|k| k.path.as_str()).collect::<HashSet<_>>().len(),
            "metal_kernels": raw.metal_names.len(),
            "metal_source_kernels": raw.metal_names.iter().filter(|n| !raw.generated_kernel_names.contains_key(*n)).count(),
            "metal_generated_kernels": raw.generated_kernel_names.len(),
            "host_files_with_dispatch": raw.dispatches.iter().map(|d| d.path.as_str()).collect::<HashSet<_>>().len(),
            "host_dispatches": raw.dispatches.len(),
            "dispatches_resolved": raw.dispatches.iter().filter(|d| !d.resolved.is_empty()).count(),
            "dispatches_unverifiable_name": raw.dispatches.iter().filter(|d| d.resolved.is_empty()).count(),
            "binding_pairs_with_matching_index_sets": raw.binding_checked,
            "threadgroup_triples_resolved": raw.geometry_checked,
            "structs_paired": raw.structs_paired,
            "referenced_kernel_names": raw.referenced.len(),
            "unreferenced_kernel_names": raw.metal_names.iter().filter(|n| !raw.referenced.contains_key(*n)).count(),
            "honesty": "PASS-like INFO is only recorded when the checker followed both sides to a concrete index or layout. A bind it could not follow is UNVERIFIABLE, never PASS. Matching indices are not a speed claim.",
            "shaders_not_in_metalcontext_library": raw.shaders_not_in_library,
        },
        "counts": {
            "ERROR": raw.counts.get("ERROR").copied().unwrap_or(0),
            "WARNING": raw.counts.get("WARNING").copied().unwrap_or(0),
            "UNVERIFIABLE": raw.counts.get("UNVERIFIABLE").copied().unwrap_or(0),
            "INFO_pass_like": raw.counts.get("INFO").copied().unwrap_or(0),
        },
        "would_waste_a_protected_window": !errors.is_empty(),
        "blocking_defect_count": errors.len(),
        "findings": trim_findings(&raw.findings),
        "queue_identity": raw.queue_identity,
        "library_membership_files": membership_files,
        "decode_family_named_kernels": raw.family_named,
        "generated_kernel_names": generated_json,
        "apple_static_limits": {
            "max_threads_per_threadgroup": APPLE_MAX_THREADS_PER_THREADGROUP,
            "simdgroup_width": APPLE_SIMDGROUP_WIDTH,
            "note": "Device hard ceiling, not a measured PSO limit. A kernel may refuse a legal-looking (256) threadgroup at pipeline creation; that is runtime."
        }
    })
}

fn collect_files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_files(&p, ext, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(p);
        }
    }
}

pub fn load_repo_sources(
    repo: &Path,
) -> (
    BTreeMap<String, String>,
    BTreeMap<String, String>,
    BTreeMap<String, String>,
) {
    let mut metal = BTreeMap::new();
    let shaders = repo.join(SHADER_DIR);
    if shaders.is_dir() {
        let mut files: Vec<PathBuf> = fs::read_dir(&shaders)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
            .collect();
        files.sort();
        for p in files {
            if let Ok(rel) = p.strip_prefix(repo) {
                if let Ok(text) = fs::read_to_string(&p) {
                    metal.insert(rel.to_string_lossy().replace('\\', "/"), text);
                }
            }
        }
    }
    let mut rust = BTreeMap::new();
    let host = repo.join(HOST_ROOT);
    if host.is_dir() {
        let mut files = Vec::new();
        collect_files(&host, "rs", &mut files);
        files.sort();
        for p in files {
            if let Ok(rel) = p.strip_prefix(repo) {
                if let Ok(text) = fs::read_to_string(&p) {
                    rust.insert(rel.to_string_lossy().replace('\\', "/"), text);
                }
            }
        }
    }
    let mut membership = BTreeMap::new();
    let modp = repo.join("crates/hawking-core/src/metal/mod.rs");
    if modp.is_file() {
        if let Ok(text) = fs::read_to_string(&modp) {
            membership = parse_library_membership(&text);
        }
    }
    (metal, rust, membership)
}

pub fn scan_repo(repo: &Path) -> Value {
    let (metal, rust, membership) = load_repo_sources(repo);
    let raw = analyze(&metal, &rust, Some(&membership), PRODUCTION_HOST_PREFIX);
    report_from_analyze(&raw, repo)
}

/// Analyze caller-supplied sources (parity with Python `analyze()`).
pub fn analyze_maps(
    metal: BTreeMap<String, String>,
    rust: BTreeMap<String, String>,
    membership: Option<BTreeMap<String, String>>,
) -> AnalyzeRaw {
    analyze(&metal, &rust, membership.as_ref(), PRODUCTION_HOST_PREFIX)
}

impl AnalyzeRaw {
    pub fn findings(&self) -> &[Finding] {
        &self.findings
    }
    pub fn binding_checked(&self) -> i64 {
        self.binding_checked
    }
    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }
    pub fn generated_kernel_names(&self) -> &BTreeMap<String, BTreeMap<String, String>> {
        &self.generated_kernel_names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct ArgbufN { uint n; float eps; };
kernel void demo_k(
    device const float* x [[buffer(0)]],
    device float* out     [[buffer(1)]],
    constant uint& n      [[buffer(2)]],
    constant float& eps   [[buffer(3)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]]
) {}
"#;

    const GOOD_HOST: &str = r#"
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    const TG: u32 = 64;
    ctx.dispatch_threads("demo_k", (64, 1, 1), (TG, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(2, n);
        enc.set_f32(3, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"#;

    const OFF_BY_ONE_HOST: &str = r#"
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(3, n);
        enc.set_f32(4, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"#;

    fn metal_map(src: &str) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("synth.metal".into(), src.into());
        m
    }
    fn rust_map(src: &str) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("synth.rs".into(), src.into());
        m
    }
    fn find<'a>(
        raw: &'a AnalyzeRaw,
        sev: &str,
        check: &str,
        kernel: Option<&str>,
    ) -> Vec<&'a Finding> {
        raw.findings
            .iter()
            .filter(|f| {
                f.severity == sev
                    && f.check == check
                    && (kernel.is_none() || f.kernel.as_deref() == kernel)
            })
            .collect()
    }

    #[test]
    fn good_pair_matching_indices_is_not_an_error() {
        let raw = analyze_maps(metal_map(GOOD_METAL), rust_map(GOOD_HOST), None);
        assert!(find(&raw, "ERROR", "binding_index", None).is_empty());
        assert!(find(&raw, "ERROR", "kernel_existence", None).is_empty());
        assert!(raw.binding_checked >= 1);
    }

    #[test]
    fn negative_control_wrong_buffer_index_is_refused() {
        let raw = analyze_maps(metal_map(GOOD_METAL), rust_map(OFF_BY_ONE_HOST), None);
        let hits = find(&raw, "ERROR", "binding_index", Some("demo_k"));
        assert!(!hits.is_empty(), "NEGATIVE CONTROL FAILED");
        let extra = &hits[0].extra;
        let off = extra
            .get("off_by_one")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(off || extra.get("extra_on_host").is_some());
    }

    #[test]
    fn missing_kernel_name_is_error() {
        let host = r#"
fn go(ctx: &MetalContext) {
    ctx.dispatch_threads("no_such_kernel", (1, 1, 1), (1, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
    });
}
"#;
        let raw = analyze_maps(metal_map(GOOD_METAL), rust_map(host), None);
        assert!(!find(&raw, "ERROR", "kernel_existence", Some("no_such_kernel")).is_empty());
    }

    #[test]
    fn type_width_mismatch_is_error() {
        let host = r#"
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |enc| {
        enc.set_u32(0, n);
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(2, n);
        enc.set_f32(3, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"#;
        let raw = analyze_maps(metal_map(GOOD_METAL), rust_map(host), None);
        let hits = find(&raw, "ERROR", "type_width", Some("demo_k"));
        assert!(!hits.is_empty());
        assert!(hits[0].message.contains("buffer(0)"));
    }

    #[test]
    fn threadgroup_over_device_limit_is_error() {
        let host = r#"
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (2048, 1, 1), (2048, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(2, n);
        enc.set_f32(3, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"#;
        let raw = analyze_maps(metal_map(GOOD_METAL), rust_map(host), None);
        let hits = find(&raw, "ERROR", "dispatch_geometry", Some("demo_k"));
        assert!(!hits.is_empty());
        assert!(hits[0].message.contains("1024"));
    }

    #[test]
    fn repr_c_field_order_mismatch_is_error() {
        let metal = r#"
#include <metal_stdlib>
using namespace metal;
struct Pack { uint a; float b; uint c; };
kernel void pack_k(constant Pack& args [[buffer(0)]]) {}
"#;
        let host = r#"
#[repr(C)]
pub struct Pack {
    pub b: f32,
    pub a: u32,
    pub c: u32,
}
fn go(ctx: &MetalContext) {
    ctx.dispatch_threads("pack_k", (1, 1, 1), (1, 1, 1), |enc| {
        enc.set_buffer(0, Some(ab.handle()), 0);
    });
}
"#;
        let raw = analyze_maps(metal_map(metal), rust_map(host), None);
        let hits = find(&raw, "ERROR", "host_shader_abi", None);
        assert!(!hits.is_empty());
        assert!(hits[0].message.contains("Pack"));
    }

    #[test]
    fn plumbing_dispatch_threads_definitions_are_skipped() {
        let plumbing = r#"
        pub fn dispatch_threads(
            &self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            let pipe = self.pipeline(fn_name)?;
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, 1, 1),
                MTLSize::new(tg.0 as u64, 1, 1),
            );
            encode(enc);
            Ok(())
        }
    "#;
        let mut rust = BTreeMap::new();
        rust.insert("metal/mod.rs".into(), plumbing.into());
        let raw = analyze_maps(metal_map(GOOD_METAL), rust, None);
        assert_eq!(raw.dispatches.len(), 0);
    }
}
