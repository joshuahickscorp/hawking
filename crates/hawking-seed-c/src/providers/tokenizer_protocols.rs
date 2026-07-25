//! **The one tokenizer/protocol registry.** A chat protocol is a *declarative descriptor*: role
//! vocabulary, turn structure, control tokens, thinking/reasoning channel semantics, tool-call encoding,
//! and multimodal content-part shape. There is exactly ONE prompt-assembly path ([`Protocol::assemble`])
//! and it is driven entirely by the descriptor. Per-provider bespoke formatting functions are forbidden by
//! the campaign, so this module contains none: every protocol below is pure data.
//!
//! ## Provenance discipline
//!
//! Every descriptor carries a [`Protocol::provenance`] string naming where its semantics came from and an
//! [`Protocol::unverified`] list naming each field that is NOT established from an official
//! `tokenizer_config.json` / chat template available to this tree. Where a control token could not be
//! attested it is left **empty**, and [`Protocol::assemble`] REFUSES rather than emitting a guess: an
//! invented control token is worse than an absent one.
//!
//! Two descriptors are attested against files on this machine:
//! - `harmony` from `models/gpt-oss-120b/tokenizer_config.json` (`added_tokens_decoder` 199998..200018),
//! - `qwen3` from `models/qwen3-235b-a22b/tokenizer_config.json` (`added_tokens_decoder` + `chat_template`).
//!
//! The rest are declared from published tokenizer_config conventions and are marked as such; anything the
//! author could not attest at all is absent, not invented.

use crate::evidence::receipt;
use crate::record::Record;
use crate::Result;
use serde::{Deserialize, Serialize};

fn err(msg: String) -> crate::Error {
    crate::Error::Tokenizer(msg)
}

/// What a protocol supports. Assembly of an unsupported turn kind is REFUSED, never silently degraded.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Capabilities {
    pub thinking_channel: bool,
    pub tool_calls: bool,
    pub multimodal: bool,
    pub system_role: bool,
    pub developer_role: bool,
}

/// The control/special token vocabulary. An EMPTY string means "not attested"; if a template references
/// it, assembly refuses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tokens {
    pub bos: String,
    pub eos: String,
    /// Opens a turn (`<|im_start|>`, `<|start|>`, `<start_of_turn>`).
    pub turn_start: String,
    /// Terminates the role field (`\n`, `<|end_header_id|>\n\n`).
    pub role_end: String,
    /// Separates header from message body (`<|message|>` in Harmony; empty elsewhere).
    pub message_sep: String,
    /// Introduces a channel name (`<|channel|>` in Harmony).
    pub channel_sep: String,
    /// Closes a turn (`<|im_end|>`, `<|end|>`, `<end_of_turn>`, `<|eot_id|>`).
    pub turn_end: String,
    /// Text between turns (usually `\n`).
    pub turn_sep: String,
    /// Inline thinking delimiters, used when the protocol has no named channel.
    pub think_open: String,
    pub think_close: String,
    /// Tokens that end generation.
    pub stop: Vec<String>,
}

/// Tool-call and tool-response encoding. All templates take `{name}` and `{arguments}` / `{content}`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolEncoding {
    /// Wraps the whole call section within the assistant turn.
    pub open: String,
    pub close: String,
    /// One call. `{name}`, `{arguments}`.
    pub payload_template: String,
    /// When non-empty, the recipient rides in the ROLE field (Harmony), e.g. `assistant to=functions.{name}`.
    pub recipient_template: String,
    /// Channel a call is emitted on (Harmony `commentary`); empty when the protocol has no channels.
    pub channel: String,
    /// Turn terminator override for a call turn (Harmony `<|call|>`); empty = use `Tokens::turn_end`.
    pub terminator: String,
    /// Role a tool result is delivered under. `{name}` allowed (Harmony `functions.{name}`).
    pub response_role: String,
    pub response_open: String,
    pub response_close: String,
}

/// The multimodal content-part shape: how a non-text part is rendered into the token stream.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultimodalShape {
    /// `{text}`
    pub text_template: String,
    /// Placeholder emitted for one image (e.g. `<|vision_start|><|image_pad|><|vision_end|>`).
    pub image_template: String,
    pub video_template: String,
    pub audio_template: String,
    /// True when the API-level message content is a JSON list of `{"type": ...}` parts.
    pub parts_are_json: bool,
}

/// A named protocol mode (thinking on/off, reasoning effort). `prefix` is injected after the generation
/// prompt; modes that are a system-message CONTENT convention rather than a token carry an empty prefix
/// and say so in `note`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMode {
    pub name: String,
    pub prefix: String,
    pub note: String,
    pub verified: bool,
}

impl ProtocolMode {
    fn new(name: &str, prefix: &str, note: &str, verified: bool) -> Self {
        ProtocolMode { name: name.into(), prefix: prefix.into(), note: note.into(), verified }
    }
}

/// One declarative protocol descriptor. This is the ENTIRE per-protocol surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Protocol {
    pub name: String,
    /// Tokenizer identity this protocol binds to (matches `SourceRecord::tokenizer_identity`).
    pub tokenizer: String,
    /// The role vocabulary. A role outside this set is refused.
    pub roles: Vec<String>,
    /// Canonical role renames (e.g. gemma `assistant` -> `model`).
    pub role_alias: Vec<(String, String)>,
    pub tokens: Tokens,
    /// Literal text prepended once to a prompt (BOS, `[gMASK]<sop>`).
    pub prefix: String,
    /// The ONE turn template. Placeholders: `{bos} {eos} {turn_start} {role} {role_end} {channel}
    /// {message} {content} {turn_end} {turn_sep}`. `{content}` is substituted last.
    pub turn_template: String,
    /// Per-role template overrides for protocols whose turns are not uniform (Mistral, DeepSeek, GLM, Phi).
    /// Still rendered by the ONE renderer; this is data, not a bespoke formatter.
    pub role_templates: Vec<(String, String)>,
    /// Channel name carried by an assistant answer turn (Harmony `final`); empty = no channels.
    pub final_channel: String,
    /// Channel name carried by a thinking turn (Harmony `analysis`); empty = use `think_open/close`.
    pub thinking_channel: String,
    pub generation_prompt: String,
    pub modes: Vec<ProtocolMode>,
    pub tools: ToolEncoding,
    pub multimodal: Option<MultimodalShape>,
    pub caps: Capabilities,
    /// Where these semantics came from.
    pub provenance: String,
    /// Fields NOT established from an official tokenizer_config / chat template available here.
    pub unverified: Vec<String>,
}

/// One content part of a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    Text(String),
    Image,
    Video,
    Audio,
}

/// A turn to assemble. The renderer is the same for every kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Turn {
    Chat { role: String, content: String },
    Thinking { content: String },
    ToolCall { name: String, arguments: String },
    ToolResponse { name: String, content: String },
    Multimodal { role: String, parts: Vec<ContentPart> },
}

impl Protocol {
    fn alias(&self, role: &str) -> String {
        self.role_alias
            .iter()
            .find(|(from, _)| from == role)
            .map(|(_, to)| to.clone())
            .unwrap_or_else(|| role.to_string())
    }

    fn template_for(&self, role: &str) -> &str {
        self.role_templates
            .iter()
            .find(|(r, _)| r == role)
            .map(|(_, t)| t.as_str())
            .unwrap_or(&self.turn_template)
    }

    fn check_role(&self, role: &str) -> Result<()> {
        if !self.roles.iter().any(|r| r == role) {
            return Err(err(format!("{}: role `{role}` is not in the declared role vocabulary {:?}", self.name, self.roles)));
        }
        if role == "system" && !self.caps.system_role {
            return Err(err(format!("{}: no system role; refusing rather than folding it into another role", self.name)));
        }
        if role == "developer" && !self.caps.developer_role {
            return Err(err(format!("{}: no developer role", self.name)));
        }
        Ok(())
    }

    /// The ONE renderer. Token-backed placeholders that are referenced but empty are REFUSED (absent or
    /// UNVERIFIED control token). `{content}` is substituted last so message text can contain braces.
    fn render(&self, template: &str, role: &str, channel: &str, content: &str, terminator: &str) -> Result<String> {
        if template.is_empty() {
            return Err(err(format!("{}: turn template is UNVERIFIED/absent for role `{role}`", self.name)));
        }
        let channel_frag = if channel.is_empty() {
            String::new()
        } else {
            if self.tokens.channel_sep.is_empty() {
                return Err(err(format!("{}: channel `{channel}` requested but channel_sep is UNVERIFIED/absent", self.name)));
            }
            format!("{}{}", self.tokens.channel_sep, channel)
        };
        let turn_end = if terminator.is_empty() { self.tokens.turn_end.as_str() } else { terminator };
        let table: [(&str, &str); 9] = [
            ("{bos}", &self.tokens.bos),
            ("{eos}", &self.tokens.eos),
            ("{turn_start}", &self.tokens.turn_start),
            ("{role_end}", &self.tokens.role_end),
            ("{message}", &self.tokens.message_sep),
            ("{turn_end}", turn_end),
            ("{turn_sep}", &self.tokens.turn_sep),
            ("{channel}", &channel_frag),
            ("{role}", role),
        ];
        let mut out = template.to_string();
        for (ph, val) in table {
            if out.contains(ph) && val.is_empty() && ph != "{channel}" {
                return Err(err(format!("{}: template needs {ph} but it is UNVERIFIED/absent", self.name)));
            }
            out = out.replace(ph, val);
        }
        Ok(out.replace("{content}", content))
    }

    fn render_role_turn(&self, role: &str, channel: &str, content: &str, terminator: &str) -> Result<String> {
        self.check_role(role)?;
        let r = self.alias(role);
        let t = self.template_for(&r).to_string();
        self.render(&t, &r, channel, content, terminator)
    }

    fn assistant_channel(&self) -> &str {
        &self.final_channel
    }

    fn render_turn(&self, turn: &Turn) -> Result<String> {
        match turn {
            Turn::Chat { role, content } => {
                let ch = if role == "assistant" { self.assistant_channel() } else { "" };
                self.render_role_turn(role, ch, content, "")
            }
            Turn::Thinking { content } => {
                if !self.caps.thinking_channel {
                    return Err(err(format!("{}: no thinking channel; refusing to assemble a thinking turn", self.name)));
                }
                if !self.thinking_channel.is_empty() {
                    let ch = self.thinking_channel.clone();
                    return self.render_role_turn("assistant", &ch, content, "");
                }
                if self.tokens.think_open.is_empty() || self.tokens.think_close.is_empty() {
                    return Err(err(format!("{}: thinking delimiters are UNVERIFIED/absent", self.name)));
                }
                let body = format!("{}{}{}", self.tokens.think_open, content, self.tokens.think_close);
                self.render_role_turn("assistant", "", &body, "")
            }
            Turn::ToolCall { name, arguments } => {
                if !self.caps.tool_calls {
                    return Err(err(format!("{}: no tool-call encoding; refusing to assemble a tool-call turn", self.name)));
                }
                if self.tools.payload_template.is_empty() {
                    return Err(err(format!("{}: tool-call payload template is UNVERIFIED/absent", self.name)));
                }
                let payload = self
                    .tools
                    .payload_template
                    .replace("{name}", name)
                    .replace("{arguments}", arguments);
                let body = format!("{}{}{}", self.tools.open, payload, self.tools.close);
                let role = if self.tools.recipient_template.is_empty() {
                    "assistant".to_string()
                } else {
                    self.tools.recipient_template.replace("{name}", name)
                };
                // A recipient-bearing role is a derived role, so it bypasses the vocabulary check but not
                // the renderer: it still goes through the ONE path.
                if self.tools.recipient_template.is_empty() {
                    self.render_role_turn("assistant", &self.tools.channel.clone(), &body, &self.tools.terminator.clone())
                } else {
                    let t = self.template_for("assistant").to_string();
                    self.render(&t, &role, &self.tools.channel, &body, &self.tools.terminator)
                }
            }
            Turn::ToolResponse { name, content } => {
                if !self.caps.tool_calls {
                    return Err(err(format!("{}: no tool-call encoding; refusing to assemble a tool-response turn", self.name)));
                }
                if self.tools.response_role.is_empty() {
                    return Err(err(format!("{}: tool-response role is UNVERIFIED/absent", self.name)));
                }
                let body = format!("{}{}{}", self.tools.response_open, content, self.tools.response_close);
                let role = self.tools.response_role.replace("{name}", name);
                if self.tools.response_role.contains("{name}") {
                    let t = self.template_for("tool").to_string();
                    self.render(&t, &role, "", &body, "")
                } else {
                    self.render_role_turn(&role, "", &body, "")
                }
            }
            Turn::Multimodal { role, parts } => {
                let shape = self.multimodal.as_ref().filter(|_| self.caps.multimodal).ok_or_else(|| {
                    err(format!("{}: no multimodal content parts; refusing to assemble a multimodal turn", self.name))
                })?;
                let mut body = String::new();
                for p in parts {
                    let (tmpl, what) = match p {
                        ContentPart::Text(_) => (&shape.text_template, "text"),
                        ContentPart::Image => (&shape.image_template, "image"),
                        ContentPart::Video => (&shape.video_template, "video"),
                        ContentPart::Audio => (&shape.audio_template, "audio"),
                    };
                    if tmpl.is_empty() {
                        return Err(err(format!("{}: {what} content part shape is UNVERIFIED/absent", self.name)));
                    }
                    match p {
                        ContentPart::Text(t) => body.push_str(&tmpl.replace("{text}", t)),
                        _ => body.push_str(tmpl),
                    }
                }
                let ch = if role == "assistant" { self.assistant_channel() } else { "" };
                self.render_role_turn(role, ch, &body, "")
            }
        }
    }

    /// The ONE prompt-assembly path. `generate_as` names a declared mode and appends the generation
    /// prompt plus that mode's prefix; `None` assembles history only.
    pub fn assemble(&self, turns: &[Turn], generate_as: Option<&str>) -> Result<String> {
        let mut out = self.prefix.clone();
        for t in turns {
            out.push_str(&self.render_turn(t)?);
        }
        if let Some(mode) = generate_as {
            if self.generation_prompt.is_empty() {
                return Err(err(format!("{}: generation prompt is UNVERIFIED/absent", self.name)));
            }
            let m = self
                .modes
                .iter()
                .find(|m| m.name == mode)
                .ok_or_else(|| err(format!("{}: unknown mode `{mode}`; declared: {:?}", self.name, self.modes.iter().map(|m| &m.name).collect::<Vec<_>>())))?;
            out.push_str(&self.generation_prompt);
            out.push_str(&m.prefix);
        }
        Ok(out)
    }
}

// ---- The built-in protocol registry. Pure data. ----

fn base(name: &str, tokenizer: &str) -> Protocol {
    Protocol {
        name: name.into(),
        tokenizer: tokenizer.into(),
        roles: vec!["system".into(), "user".into(), "assistant".into(), "tool".into()],
        role_alias: vec![],
        tokens: Tokens::default(),
        prefix: String::new(),
        turn_template: "{turn_start}{role}{role_end}{content}{turn_end}{turn_sep}".into(),
        role_templates: vec![],
        final_channel: String::new(),
        thinking_channel: String::new(),
        generation_prompt: String::new(),
        modes: vec![ProtocolMode::new("default", "", "", true)],
        tools: ToolEncoding::default(),
        multimodal: None,
        caps: Capabilities { system_role: true, ..Capabilities::default() },
        provenance: String::new(),
        unverified: vec![],
    }
}

/// **Harmony** (gpt-oss). Attested: control tokens from the local `gpt-oss-120b/tokenizer_config.json`
/// (`<|start|>` 200006, `<|end|>` 200007, `<|message|>` 200008, `<|channel|>` 200005, `<|call|>` 200012,
/// `<|return|>` 200002, `<|constrain|>` 200003), `eos_token = <|return|>`.
pub fn harmony() -> Protocol {
    let mut p = base("harmony", "o200k_harmony");
    p.roles = vec!["system".into(), "developer".into(), "user".into(), "assistant".into(), "tool".into()];
    p.tokens = Tokens {
        bos: "<|startoftext|>".into(),
        eos: "<|return|>".into(),
        turn_start: "<|start|>".into(),
        role_end: String::new(),
        message_sep: "<|message|>".into(),
        channel_sep: "<|channel|>".into(),
        turn_end: "<|end|>".into(),
        turn_sep: String::new(),
        think_open: String::new(),
        think_close: String::new(),
        stop: vec!["<|return|>".into(), "<|call|>".into(), "<|endoftext|>".into()],
    };
    p.turn_template = "{turn_start}{role}{channel}{message}{content}{turn_end}".into();
    p.final_channel = "final".into();
    p.thinking_channel = "analysis".into();
    p.generation_prompt = "<|start|>assistant".into();
    p.modes = vec![
        ProtocolMode::new("low", "", "reasoning effort is a `Reasoning: low` line in the system message, not a token", false),
        ProtocolMode::new("medium", "", "as above", false),
        ProtocolMode::new("high", "", "as above", false),
        ProtocolMode::new("default", "", "", true),
    ];
    p.tools = ToolEncoding {
        open: String::new(),
        close: String::new(),
        payload_template: "{arguments}".into(),
        recipient_template: "assistant to=functions.{name}".into(),
        channel: "commentary".into(),
        terminator: "<|call|>".into(),
        response_role: "functions.{name} to=assistant".into(),
        response_open: String::new(),
        response_close: String::new(),
    };
    p.caps = Capabilities { thinking_channel: true, tool_calls: true, multimodal: false, system_role: true, developer_role: true };
    p.provenance = "local models/gpt-oss-120b/tokenizer_config.json added_tokens_decoder 199998..200018".into();
    p.unverified = vec![
        "system-message content grammar (Knowledge cutoff / Reasoning / Valid channels lines) is prose convention, not tokens".into(),
        "developer-message TypeScript tool-namespace body".into(),
        "<|constrain|> placement for typed tool arguments".into(),
        "reasoning-effort mode encoding (declared with empty prefix)".into(),
    ];
    p
}

/// **Qwen3** chat + thinking. Attested: `models/qwen3-235b-a22b/tokenizer_config.json`
/// (`<|im_start|>` 151644, `<|im_end|>` 151645, `<tool_call>` 151657, `</tool_call>` 151658,
/// `<tool_response>` 151665, `</tool_response>` 151666, `<think>` 151667, `</think>` 151668) and its
/// `chat_template` (turn shape, tool-call JSON payload, tool results delivered under the `user` role,
/// generation prompt).
pub fn qwen3() -> Protocol {
    let mut p = base("qwen3", "qwen2.tokenizer");
    p.tokens = Tokens {
        bos: String::new(),
        eos: "<|im_end|>".into(),
        turn_start: "<|im_start|>".into(),
        role_end: "\n".into(),
        message_sep: String::new(),
        channel_sep: String::new(),
        turn_end: "<|im_end|>".into(),
        turn_sep: "\n".into(),
        think_open: "<think>".into(),
        think_close: "</think>".into(),
        stop: vec!["<|im_end|>".into(), "<|endoftext|>".into()],
    };
    p.generation_prompt = "<|im_start|>assistant\n".into();
    p.modes = vec![
        ProtocolMode::new("thinking", "", "", true),
        ProtocolMode::new("non_thinking", "<think>\n\n</think>\n\n", "empty-think injection is from the original Qwen3 release template; the local 2507 template has no enable_thinking branch", false),
        ProtocolMode::new("default", "", "", true),
    ];
    p.tools = ToolEncoding {
        open: String::new(),
        close: String::new(),
        payload_template: "<tool_call>\n{\"name\": \"{name}\", \"arguments\": {arguments}}\n</tool_call>".into(),
        recipient_template: String::new(),
        channel: String::new(),
        terminator: String::new(),
        response_role: "user".into(),
        response_open: "\n<tool_response>\n".into(),
        response_close: "\n</tool_response>".into(),
    };
    p.caps = Capabilities { thinking_channel: true, tool_calls: true, multimodal: false, system_role: true, developer_role: false };
    p.provenance = "local models/qwen3-235b-a22b/tokenizer_config.json (added_tokens_decoder + chat_template)".into();
    p.unverified = vec![
        "thinking mode: <think>/</think> exist as tokens 151667/151668 but the local 2507 chat_template never emits them".into(),
        "non_thinking mode prefix (see mode note)".into(),
    ];
    p
}

/// **Qwen VL** multimodal. The vision tokens are attested as PRESENT in the Qwen tokenizer
/// (`<|vision_start|>` 151652, `<|vision_end|>` 151653, `<|vision_pad|>` 151654, `<|image_pad|>` 151655,
/// `<|video_pad|>` 151656); their PLACEMENT is the published Qwen2-VL convention and is marked unverified
/// here because no VL chat_template is present on this machine.
pub fn qwen_vl() -> Protocol {
    let mut p = qwen3();
    p.name = "qwen_vl".into();
    p.multimodal = Some(MultimodalShape {
        text_template: "{text}".into(),
        image_template: "<|vision_start|><|image_pad|><|vision_end|>".into(),
        video_template: "<|vision_start|><|video_pad|><|vision_end|>".into(),
        audio_template: String::new(),
        parts_are_json: true,
    });
    p.caps.multimodal = true;
    p.provenance = "vision tokens from local models/qwen3-235b-a22b/tokenizer_config.json; placement from published Qwen2-VL convention".into();
    p.unverified = vec![
        "image/video placeholder PLACEMENT (no VL chat_template on this machine)".into(),
        "per-image pad repetition count (grid-dependent)".into(),
        "audio content part: absent, not invented".into(),
    ];
    p
}

/// **DeepSeek V3** chat. Published tokenizer_config convention: `<｜begin▁of▁sentence｜>`,
/// `<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>` (fullwidth U+FF5C, U+2581), system content
/// prepended raw with no role tag.
pub fn deepseek_v3() -> Protocol {
    let mut p = base("deepseek_v3", "deepseek.tokenizer");
    p.tokens = Tokens {
        bos: "<｜begin▁of▁sentence｜>".into(),
        eos: "<｜end▁of▁sentence｜>".into(),
        turn_start: String::new(),
        role_end: String::new(),
        message_sep: String::new(),
        channel_sep: String::new(),
        turn_end: String::new(),
        turn_sep: String::new(),
        think_open: String::new(),
        think_close: String::new(),
        stop: vec!["<｜end▁of▁sentence｜>".into()],
    };
    p.prefix = "<｜begin▁of▁sentence｜>".into();
    p.turn_template = String::new();
    p.role_templates = vec![
        ("system".into(), "{content}".into()),
        ("user".into(), "<｜User｜>{content}".into()),
        ("assistant".into(), "<｜Assistant｜>{content}{eos}".into()),
        ("tool".into(), "<｜tool▁outputs▁begin｜>{content}<｜tool▁outputs▁end｜>".into()),
    ];
    p.generation_prompt = "<｜Assistant｜>".into();
    p.tools = ToolEncoding {
        open: "<｜tool▁calls▁begin｜>".into(),
        close: "<｜tool▁calls▁end｜>".into(),
        payload_template: "<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{arguments}\n```<｜tool▁call▁end｜>".into(),
        recipient_template: String::new(),
        channel: String::new(),
        terminator: String::new(),
        response_role: "tool".into(),
        response_open: "<｜tool▁output▁begin｜>".into(),
        response_close: "<｜tool▁output▁end｜>".into(),
    };
    p.caps = Capabilities { thinking_channel: false, tool_calls: true, multimodal: false, system_role: true, developer_role: false };
    p.provenance = "published DeepSeek-V3 tokenizer_config conventions; NOT confirmed against a local file".into();
    p.unverified = vec![
        "every control token here is recalled, not read from a local tokenizer_config".into(),
        "tool-call payload fence (```json) framing".into(),
        "developer role: DeepSeek declares none; not invented".into(),
    ];
    p
}

/// **DeepSeek R1** reasoning. Same turn grammar as V3 plus `<think>`/`</think>`; the R1 template opens the
/// assistant turn with `<think>` already emitted.
pub fn deepseek_r1() -> Protocol {
    let mut p = deepseek_v3();
    p.name = "deepseek_r1".into();
    p.tokens.think_open = "<think>".into();
    p.tokens.think_close = "</think>".into();
    p.caps.thinking_channel = true;
    p.modes = vec![
        ProtocolMode::new("reasoning", "<think>\n", "R1 templates prefill the opening think tag", false),
        ProtocolMode::new("default", "", "", true),
    ];
    p.provenance = "published DeepSeek-R1 tokenizer_config conventions; NOT confirmed against a local file".into();
    p.unverified = vec![
        "every control token here is recalled, not read from a local tokenizer_config".into(),
        "prefilled <think> in the generation prompt".into(),
        "tool-call encoding under reasoning mode".into(),
    ];
    p
}

/// **GLM-4 / GLM-4.5**. Published convention: `[gMASK]<sop>` prefix, `<|system|>`, `<|user|>`,
/// `<|assistant|>`, `<|observation|>` role tags. Thinking uses `<think>`/`</think>`; the thinking-EFFORT
/// mode encoding is not established here and is left absent.
pub fn glm4() -> Protocol {
    let mut p = base("glm4", "glm4.tokenizer");
    p.roles = vec!["system".into(), "user".into(), "assistant".into(), "observation".into(), "tool".into()];
    p.role_alias = vec![("tool".into(), "observation".into())];
    p.prefix = "[gMASK]<sop>".into();
    p.tokens.eos = "<|user|>".into();
    p.tokens.think_open = "<think>".into();
    p.tokens.think_close = "</think>".into();
    p.tokens.stop = vec!["<|user|>".into(), "<|observation|>".into(), "<|endoftext|>".into()];
    p.turn_template = String::new();
    p.role_templates = vec![
        ("system".into(), "<|system|>\n{content}".into()),
        ("user".into(), "<|user|>\n{content}".into()),
        ("assistant".into(), "<|assistant|>\n{content}".into()),
        ("observation".into(), "<|observation|>\n{content}".into()),
    ];
    p.generation_prompt = "<|assistant|>\n".into();
    p.modes = vec![
        ProtocolMode::new("thinking", "", "GLM thinking-effort control is NOT established as a token; absent rather than invented", false),
        ProtocolMode::new("no_thinking", "", "as above", false),
        ProtocolMode::new("default", "", "", true),
    ];
    p.tools = ToolEncoding { response_role: "observation".into(), ..ToolEncoding::default() };
    p.caps = Capabilities { thinking_channel: true, tool_calls: true, multimodal: false, system_role: true, developer_role: false };
    p.provenance = "published GLM-4 chat template convention; NOT confirmed against a local file".into();
    p.unverified = vec![
        "thinking-effort mode encoding: ABSENT (no token attested)".into(),
        "tool-call payload template: ABSENT, so tool-call assembly refuses".into(),
        "<think>/</think> presence in the GLM tokenizer vocabulary".into(),
    ];
    p
}

/// **Kimi K2** interleaved thinking + tool calls. The author cannot attest Kimi's control tokens from any
/// file on this machine, so they are ABSENT: capabilities are declared, assembly refuses. An invented
/// control token would be worse.
pub fn kimi_k2() -> Protocol {
    let mut p = base("kimi_k2", "kimi.tokenizer");
    p.turn_template = String::new();
    p.caps = Capabilities { thinking_channel: true, tool_calls: true, multimodal: false, system_role: true, developer_role: false };
    p.provenance = "capability shape only; no tokenizer_config available and no token attested".into();
    p.unverified = vec![
        "ALL control tokens: ABSENT (turn start/end, role tags)".into(),
        "interleaved thinking channel encoding: ABSENT".into(),
        "tool-call section encoding: ABSENT".into(),
        "assembly REFUSES until a tokenizer_config is hydrated".into(),
    ];
    p
}

/// **MiniMax** thinking modes. Same discipline as [`kimi_k2`]: capabilities declared, tokens absent.
pub fn minimax() -> Protocol {
    let mut p = base("minimax", "minimax.tokenizer");
    p.turn_template = String::new();
    p.caps = Capabilities { thinking_channel: true, tool_calls: true, multimodal: false, system_role: true, developer_role: false };
    p.provenance = "capability shape only; no tokenizer_config available and no token attested".into();
    p.unverified = vec![
        "ALL control tokens: ABSENT".into(),
        "thinking-mode encoding: ABSENT".into(),
        "assembly REFUSES until a tokenizer_config is hydrated".into(),
    ];
    p
}

/// **Llama 3**. Published special-token set: `<|begin_of_text|>`, `<|start_header_id|>`,
/// `<|end_header_id|>`, `<|eot_id|>`; tool calls via `<|python_tag|>` / `<|eom_id|>`.
pub fn llama3() -> Protocol {
    let mut p = base("llama3", "llama3.tokenizer");
    p.roles = vec!["system".into(), "user".into(), "assistant".into(), "ipython".into(), "tool".into()];
    p.role_alias = vec![("tool".into(), "ipython".into())];
    p.prefix = "<|begin_of_text|>".into();
    p.tokens = Tokens {
        bos: "<|begin_of_text|>".into(),
        eos: "<|eot_id|>".into(),
        turn_start: "<|start_header_id|>".into(),
        role_end: "<|end_header_id|>\n\n".into(),
        message_sep: String::new(),
        channel_sep: String::new(),
        turn_end: "<|eot_id|>".into(),
        turn_sep: String::new(),
        think_open: String::new(),
        think_close: String::new(),
        stop: vec!["<|eot_id|>".into(), "<|eom_id|>".into()],
    };
    p.generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n".into();
    p.tools = ToolEncoding {
        open: "<|python_tag|>".into(),
        close: String::new(),
        payload_template: "{\"name\": \"{name}\", \"parameters\": {arguments}}".into(),
        recipient_template: String::new(),
        channel: String::new(),
        terminator: "<|eom_id|>".into(),
        response_role: "ipython".into(),
        response_open: String::new(),
        response_close: String::new(),
    };
    p.caps = Capabilities { thinking_channel: false, tool_calls: true, multimodal: false, system_role: true, developer_role: false };
    p.provenance = "published Llama 3.1 special-token set; NOT confirmed against a local file".into();
    p.unverified = vec!["tool-call framing (<|python_tag|> + <|eom_id|>) and the JSON parameter key".into()];
    p
}

/// **Mistral** `[INST]` convention. No role tags and no system turn.
pub fn mistral() -> Protocol {
    let mut p = base("mistral", "mistral.tokenizer");
    p.roles = vec!["user".into(), "assistant".into(), "tool".into()];
    p.prefix = "<s>".into();
    p.tokens.bos = "<s>".into();
    p.tokens.eos = "</s>".into();
    p.tokens.stop = vec!["</s>".into()];
    p.turn_template = String::new();
    p.role_templates = vec![
        ("user".into(), "[INST] {content}[/INST]".into()),
        ("assistant".into(), " {content}{eos}".into()),
        ("tool".into(), "[TOOL_RESULTS] {content}[/TOOL_RESULTS]".into()),
    ];
    p.generation_prompt = " ".into();
    p.tools = ToolEncoding {
        open: "[TOOL_CALLS] ".into(),
        close: String::new(),
        payload_template: "[{\"name\": \"{name}\", \"arguments\": {arguments}}]".into(),
        recipient_template: String::new(),
        channel: String::new(),
        terminator: String::new(),
        response_role: "tool".into(),
        response_open: String::new(),
        response_close: String::new(),
    };
    p.caps = Capabilities { thinking_channel: false, tool_calls: true, multimodal: false, system_role: false, developer_role: false };
    p.provenance = "published Mistral v3 tokenizer special tokens; NOT confirmed against a local file".into();
    p.unverified = vec![
        "system-message handling (Mistral folds it into the first user turn); refused here rather than folded".into(),
        "[TOOL_CALLS] payload array shape".into(),
        "generation prompt: Mistral has no assistant header, only the space after [/INST]".into(),
    ];
    p
}

/// **Gemma**. `<start_of_turn>` / `<end_of_turn>`, roles `user` and `model` ONLY: no system role, so a
/// system turn is refused rather than folded into the user turn.
pub fn gemma() -> Protocol {
    let mut p = base("gemma", "gemma.tokenizer");
    p.roles = vec!["user".into(), "assistant".into(), "model".into()];
    p.role_alias = vec![("assistant".into(), "model".into())];
    p.prefix = "<bos>".into();
    p.tokens = Tokens {
        bos: "<bos>".into(),
        eos: "<eos>".into(),
        turn_start: "<start_of_turn>".into(),
        role_end: "\n".into(),
        message_sep: String::new(),
        channel_sep: String::new(),
        turn_end: "<end_of_turn>".into(),
        turn_sep: "\n".into(),
        think_open: String::new(),
        think_close: String::new(),
        stop: vec!["<end_of_turn>".into(), "<eos>".into()],
    };
    p.generation_prompt = "<start_of_turn>model\n".into();
    p.caps = Capabilities::default();
    p.provenance = "published Gemma chat template convention; NOT confirmed against a local file".into();
    p.unverified = vec![
        "Gemma 3 multimodal <start_of_image> shape: ABSENT here (this descriptor is text-only)".into(),
        "tool-call encoding: ABSENT (Gemma declares none in the base template)".into(),
    ];
    p
}

/// **Phi-3**. `<|system|>` / `<|user|>` / `<|assistant|>` with `<|end|>` terminators.
pub fn phi3() -> Protocol {
    let mut p = base("phi3", "phi3.tokenizer");
    p.roles = vec!["system".into(), "user".into(), "assistant".into()];
    p.tokens.bos = "<s>".into();
    p.tokens.eos = "<|endoftext|>".into();
    p.tokens.stop = vec!["<|end|>".into(), "<|endoftext|>".into()];
    p.turn_template = String::new();
    p.role_templates = vec![
        ("system".into(), "<|system|>\n{content}<|end|>\n".into()),
        ("user".into(), "<|user|>\n{content}<|end|>\n".into()),
        ("assistant".into(), "<|assistant|>\n{content}<|end|>\n".into()),
    ];
    p.generation_prompt = "<|assistant|>\n".into();
    p.caps = Capabilities { system_role: true, ..Capabilities::default() };
    p.provenance = "published Phi-3 chat template convention; NOT confirmed against a local file".into();
    p.unverified = vec!["tool-call encoding: ABSENT".into(), "Phi-4-reasoning thinking tags: not covered by this descriptor".into()];
    p
}

/// The full protocol set the nucleus ships.
pub fn builtins() -> Vec<Protocol> {
    vec![
        harmony(),
        qwen3(),
        qwen_vl(),
        deepseek_v3(),
        deepseek_r1(),
        glm4(),
        kimi_k2(),
        minimax(),
        llama3(),
        mistral(),
        gemma(),
        phi3(),
    ]
}

pub fn lookup(name: &str) -> Option<Protocol> {
    builtins().into_iter().find(|p| p.name == name)
}

/// Seal the whole registry as a `tokenizer_protocols` receipt through the Seed's ONE evidence engine, so
/// a launch packet carries the exact protocol data that assembled its prompts.
pub fn seal_registry() -> Result<Record> {
    Ok(receipt("tokenizer_protocols", serde_json::to_value(builtins())?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(s: &str) -> Turn {
        Turn::Chat { role: "user".into(), content: s.into() }
    }

    #[test]
    fn qwen_plain_chat_matches_local_chat_template() {
        let out = qwen3().assemble(&[user("Hi")], Some("default")).unwrap();
        assert_eq!(out, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn qwen_thinking_and_non_thinking_modes() {
        let p = qwen3();
        let t = p.assemble(&[Turn::Thinking { content: "step".into() }], None).unwrap();
        assert_eq!(t, "<|im_start|>assistant\n<think>step</think><|im_end|>\n");
        let nt = p.assemble(&[user("Hi")], Some("non_thinking")).unwrap();
        assert!(nt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen_tool_call_and_response() {
        let p = qwen3();
        let call = p
            .assemble(&[Turn::ToolCall { name: "get_time".into(), arguments: "{\"tz\": \"UTC\"}".into() }], None)
            .unwrap();
        assert_eq!(
            call,
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}\n</tool_call><|im_end|>\n"
        );
        let resp = p
            .assemble(&[Turn::ToolResponse { name: "get_time".into(), content: "12:00".into() }], None)
            .unwrap();
        assert_eq!(resp, "<|im_start|>user\n\n<tool_response>\n12:00\n</tool_response><|im_end|>\n");
    }

    #[test]
    fn harmony_chat_thinking_and_tool_call() {
        let p = harmony();
        let chat = p.assemble(&[user("Hi")], Some("default")).unwrap();
        assert_eq!(chat, "<|start|>user<|message|>Hi<|end|><|start|>assistant");
        let think = p.assemble(&[Turn::Thinking { content: "reason".into() }], None).unwrap();
        assert_eq!(think, "<|start|>assistant<|channel|>analysis<|message|>reason<|end|>");
        let call = p
            .assemble(&[Turn::ToolCall { name: "lookup".into(), arguments: "{\"q\":1}".into() }], None)
            .unwrap();
        assert_eq!(call, "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{\"q\":1}<|call|>");
        // an assistant answer rides the `final` channel
        let ans = p.assemble(&[Turn::Chat { role: "assistant".into(), content: "42".into() }], None).unwrap();
        assert_eq!(ans, "<|start|>assistant<|channel|>final<|message|>42<|end|>");
    }

    #[test]
    fn deepseek_chat_thinking_and_tool_call() {
        let chat = deepseek_v3().assemble(&[user("Hi")], Some("default")).unwrap();
        assert_eq!(chat, "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜>");
        let think = deepseek_r1().assemble(&[Turn::Thinking { content: "why".into() }], None).unwrap();
        assert!(think.contains("<think>why</think>"));
        let call = deepseek_v3()
            .assemble(&[Turn::ToolCall { name: "f".into(), arguments: "{}".into() }], None)
            .unwrap();
        assert!(call.contains("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>f"));
    }

    #[test]
    fn multimodal_turn_assembles_only_where_declared() {
        let parts = vec![ContentPart::Image, ContentPart::Text("what is this".into())];
        let out = qwen_vl()
            .assemble(&[Turn::Multimodal { role: "user".into(), parts: parts.clone() }], None)
            .unwrap();
        assert_eq!(out, "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>what is this<|im_end|>\n");
        // text-only Qwen declares no multimodal capability and REFUSES
        let refused = qwen3().assemble(&[Turn::Multimodal { role: "user".into(), parts }], None);
        assert!(refused.is_err(), "text-only protocol must refuse a multimodal turn");
    }

    #[test]
    fn missing_capability_refuses_instead_of_degrading() {
        // no thinking channel
        assert!(llama3().assemble(&[Turn::Thinking { content: "x".into() }], None).is_err());
        assert!(gemma().assemble(&[Turn::Thinking { content: "x".into() }], None).is_err());
        // no tool calls
        assert!(gemma().assemble(&[Turn::ToolCall { name: "f".into(), arguments: "{}".into() }], None).is_err());
        assert!(phi3().assemble(&[Turn::ToolCall { name: "f".into(), arguments: "{}".into() }], None).is_err());
        // no system role
        assert!(mistral().assemble(&[Turn::Chat { role: "system".into(), content: "s".into() }], None).is_err());
        assert!(gemma().assemble(&[Turn::Chat { role: "system".into(), content: "s".into() }], None).is_err());
        // no developer role
        assert!(qwen3().assemble(&[Turn::Chat { role: "developer".into(), content: "d".into() }], None).is_err());
        assert!(harmony().assemble(&[Turn::Chat { role: "developer".into(), content: "d".into() }], None).is_ok());
    }

    #[test]
    fn unverified_tokens_refuse_rather_than_guess() {
        // Kimi and MiniMax declare capabilities but no attested tokens: every turn kind refuses.
        for p in [kimi_k2(), minimax()] {
            assert!(p.assemble(&[user("Hi")], None).is_err(), "{} must refuse", p.name);
            assert!(p.assemble(&[Turn::Thinking { content: "t".into() }], None).is_err());
            assert!(p.assemble(&[Turn::ToolCall { name: "f".into(), arguments: "{}".into() }], None).is_err());
            assert!(!p.unverified.is_empty());
        }
        // GLM has turn tokens but no attested tool-call payload: chat works, tool call refuses.
        let glm = glm4();
        assert!(glm.assemble(&[user("Hi")], Some("default")).is_ok());
        assert!(glm.assemble(&[Turn::ToolCall { name: "f".into(), arguments: "{}".into() }], None).is_err());
    }

    #[test]
    fn unknown_role_and_unknown_mode_are_refused() {
        assert!(qwen3().assemble(&[Turn::Chat { role: "narrator".into(), content: "x".into() }], None).is_err());
        assert!(qwen3().assemble(&[user("Hi")], Some("turbo")).is_err());
    }

    #[test]
    fn every_protocol_declares_provenance_and_seals_as_json() {
        for p in builtins() {
            assert!(!p.provenance.is_empty(), "{} has no provenance", p.name);
            assert!(!p.roles.is_empty(), "{} has no role vocabulary", p.name);
            let round: Protocol = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(round.name, p.name);
        }
        let rec = seal_registry().unwrap();
        assert!(rec.verify().is_ok() && rec.kind == "tokenizer_protocols");
        assert!(lookup("harmony").is_some() && lookup("nope").is_none());
    }
}
